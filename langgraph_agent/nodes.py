import json
import logging
from typing import Annotated, Literal
from langchain_core.messages import AIMessage, HumanMessage,  SystemMessage, ToolMessage
from langgraph.types import Command, interrupt
from langchain_openai import ChatOpenAI
from state import State
from prompts import *
from tools import *
# os.environ["DEEPSEEK_API_KEY"] = ""
from langchain_deepseek import ChatDeepSeek
# llm = ChatDeepSeek(model="deepseek-chat")
llm = ChatOpenAI(model="qwen-plus-2025-04-28", temperature=0.0, base_url='https://dashscope.aliyuncs.com/compatible-mode/v1', api_key='')
# llm = ChatOpenAI(model="deepseek-chat", temperature=0.0, base_url='https://api.deepseek.com', api_key='')

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
hander = logging.StreamHandler()
hander.setLevel(logging.INFO)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
hander.setFormatter(formatter)
logger.addHandler(hander)

def extract_json(text):
    if '```json' not in text:
        return text
    text = text.split('```json')[1].split('```')[0].strip()
    return text

def extract_answer(text):
    if '</think>' in text:
        answer = text.split("</think>")[-1]
        return answer.strip()
    
    return text

def create_planner_node(state: State):
    logger.info("***正在运行Create Planner node***")
    messages = [SystemMessage(content=PLAN_SYSTEM_PROMPT), HumanMessage(content=PLAN_CREATE_PROMPT.format(user_message = state['user_message']))]
    response = llm.invoke(messages)
    response = response.model_dump_json(indent=4, exclude_none=True)
    response = json.loads(response)
    plan = json.loads(extract_json(extract_answer(response['content'])))
    state['messages'] += [AIMessage(content=json.dumps(plan, ensure_ascii=False))]
    return Command(goto="execute", update={"plan": plan})

def update_planner_node(state: State):
    logger.info("***正在运行Update Planner node***")
    plan = state['plan']
    goal = plan['goal']
    state['messages'].extend([SystemMessage(content=PLAN_SYSTEM_PROMPT), HumanMessage(content=UPDATE_PLAN_PROMPT.format(plan = plan, goal=goal))])
    messages = state['messages']
    while True:
        try:
            response = llm.invoke(messages)
            response = response.model_dump_json(indent=4, exclude_none=True)
            response = json.loads(response)
            plan = json.loads(extract_json(extract_answer(response['content'])))
            state['messages']+=[AIMessage(content=json.dumps(plan, ensure_ascii=False))]
            return Command(goto="execute", update={"plan": plan})
        except Exception as e:
            messages += [HumanMessage(content=f"json格式错误:{e}")]
    
def execute_node(state: State):
    """
    LangGraph 执行节点：自动选择工具→执行→反思。

    设计要点（针对同步 graph.invoke 调用）：
    1. **反思阶段检测**：若上一轮已插入 AIMessage(tool_calls)+ToolMessage，则立即让模型反思；
       直到反思回复中不再包含 `tool_calls`。
    2. **初始阶段**：为当前待办步骤构造 System+Human 提示，
       第一次调用模型决定工具；若有工具调用，则执行并把 ToolMessage
       与 AIMessage(tool_calls) 一并写入 `state["messages"]`，然后 return，
       等下一轮进入反思阶段。
    3. 任何时候只要模型回复不带 tool_calls，直接标记步骤完成。

    "思路图"  
    初始 → AI(tool_calls) → ToolMessage → (返回 execute)
       ↘ 无 tool_calls
          ↘ 标记完成 / 进入下一步
    反思 → AI(no tool_calls) → 标记完成 / 进入下一步
    """

    # ============================= 0. 记录启动 =============================
    logger.info("***正在运行execute_node***")

    # 复制历史消息，后续可能修改
    messages = list(state.get("messages", []))

    # =====================================================================
    # 1. 反思阶段：若末尾是  AIMessage(tool_calls) + ToolMessage，就让模型反思
    # =====================================================================
    if (
        len(messages) >= 2
        and isinstance(messages[-2], AIMessage)
        and messages[-2].additional_kwargs.get("tool_calls")
        and isinstance(messages[-1], ToolMessage)
    ):
        logger.info("🔁 进入工具反思阶段")

        # 可能多轮反思：只要模型继续返回 tool_calls，就继续执行工具
        while True:
            # 反思调用必须绑定工具（模型可能再次调用）
            MAX_REFLECT = 3 #模型最多尝试3次
            tries = 0 # 尝试次数
            reflect_resp = llm.bind_tools([create_file, str_replace, shell_exec]).invoke(messages)
            logger.info(f"♻️ 反思返回:\n{reflect_resp}")   # 把反思内容打出来
            # 若不再带 tool_calls，则反思结束
            if not reflect_resp.tool_calls:
                messages.append(reflect_resp)
                break
            tries += 1

            # 超过次数尚未成功
            if tries == MAX_REFLECT:
                logger.warning("⚠️ 反思已达到最大重试次数，仍未成功，跳过该步骤。")
                steps[idx]['status'] = 'skipped'

            # 否则执行新工具，再把结果作为 ToolMessage 写入
            tc = reflect_resp.tool_calls[0]
            tc_id, tname, targs = tc["id"], tc["name"], tc["args"]
            tool_map = {"create_file": create_file, "str_replace": str_replace, "shell_exec": shell_exec}
            tresult = tool_map[tname].invoke(targs) if tname in tool_map else {"error": "unknown tool"}
            tool_msg = ToolMessage(content=json.dumps(tresult, ensure_ascii=False), tool_call_id=tc_id)
            messages.extend([reflect_resp, tool_msg])  # 继续循环反思
            logger.info("工具调用 JSON: %s", json.dumps(tc, ensure_ascii=False, indent=2))
            

        # ------- 步骤标记为完成并写回 state -------
        plan = state.get("plan", {})
        steps = plan.get("steps", []) if isinstance(plan, dict) else plan
        idx = next((i for i, s in enumerate(steps) if s.get("status") == "pending"), None)
        if idx is not None:
            steps[idx]["status"] = "completed"
        # past_steps 记录
        past_steps = state.get("past_steps", [])
        past_steps.append((f"步骤 {idx+1}: {steps[idx]['title']}", messages[-1].content))

        return Command(goto="execute", update={
            "plan": plan,
            "messages": messages,
            "past_steps": past_steps
        })

    # =====================================================================
    # 2. 初始阶段：正常选择工具并执行
    # =====================================================================
    plan_obj = state.get("plan", {})
    steps = plan_obj.get("steps", []) if isinstance(plan_obj, dict) else plan_obj

    # 找到第一个 pending 步骤
    cur_task, task_idx = None, -1
    for i, st in enumerate(steps):
        if st.get("status") == "pending":
            cur_task, task_idx = st, i
            break

    # 所有步骤完成 → 跳到 report
    if cur_task is None:
        logger.info("所有任务已完成，准备生成报告。")
        summary = "\n\n---\n\n".join([
            f"## {t}\n\n**结果:**\n```\n{r}\n```" for t, r in state.get("past_steps", [])
        ])
        obs_msg = [HumanMessage(content="这是之前所有步骤的执行摘要和结果，请根据这些信息生成最终报告：\n\n"+summary)]
        return Command(goto="report", update={"observations": obs_msg})

    logger.info(f"当前执行STEP:{cur_task}")

    # -------- 构造 prompt --------
    if not any(isinstance(m, SystemMessage) and m.content == EXECUTE_SYSTEM_PROMPT for m in messages):
        messages.append(SystemMessage(content=EXECUTE_SYSTEM_PROMPT))

    messages.append(HumanMessage(content=EXECUTION_PROMPT.format(
        user_message=state.get('user_message', ''), step=cur_task)))

    # -------- 第一次模型调用：选择并调用工具 --------
    first_resp = llm.bind_tools([create_file, str_replace, shell_exec]).invoke(messages)
    logger.info(f"🛠️ 首次调用返回:\n{first_resp}")

    # 若模型直接完成，无 tool_calls
    if not first_resp.tool_calls:
        steps[task_idx]['status'] = 'completed'
        return Command(goto="execute", update={
            "plan": plan_obj,
            "messages": messages + [first_resp],
        })

    # ---- 执行工具 ----
    tc = first_resp.tool_calls[0]
    tname, targs, tc_id = tc["name"], tc["args"], tc["id"]
    tool_map = {"create_file": create_file, "str_replace": str_replace, "shell_exec": shell_exec}
    tresult = tool_map[tname].invoke(targs) if tname in tool_map else {"error": "unknown tool"}
    logger.info(f"已执行工具 {tname}，结果: {tresult}")

    # 记录 AI(tool_calls)+ToolMessage，并返回执行节点等待反思
    tool_msg = ToolMessage(content=json.dumps(tresult, ensure_ascii=False), tool_call_id=tc_id)
    messages.extend([first_resp, tool_msg])

    return Command(goto="execute", update={
        "plan": plan_obj,
        "messages": messages,
        "past_steps": state.get("past_steps", [])
    })


def report_node(state: State):
    """根据执行摘要，生成并保存最终报告"""
    logger.info("***正在运行report_node***")
    
    # 从状态中获取由 execute_node 准备好的、包含上下文信息的 observations
    observation_messages = state.get("observations", [HumanMessage(content="没有可用的观察结果。")])

    # 准备调用大模型所需的消息列表
    messages = observation_messages + [SystemMessage(content=REPORT_SYSTEM_PROMPT)]
    
    # 调用大模型生成报告内容，它可能会使用 create_file 工具
    response = llm.bind_tools([create_file]).invoke(messages)

    # 检查模型是否请求调用工具（即保存文件）
    if response.tool_calls:
        tool_call = response.tool_calls[0]
        if tool_call.get("name") == create_file.name:
            logger.info("模型请求创建报告文件。")
            report_content = tool_call.get("args", {}).get("file_contents", "")
            tool_result = create_file.invoke(tool_call.get("args"))
            logger.info(f"tool_result: {tool_result}")
            final_report_content = report_content
            # 修复对返回结构的处理
            if isinstance(tool_result.get("message"), str):
                logger.info("工具结果: %s", tool_result["message"])
            elif isinstance(tool_result.get("message"), dict):
                logger.info("工具结果 stdout▼\n%s\nstderr▼\n%s",
                            tool_result["message"].get("stdout", ""),
                            tool_result["message"].get("stderr", ""))
            else:
                logger.info("工具返回未知结构: %s", tool_result)

            final_report_content = report_content
        else:
            final_report_content = f"报告生成期间出现意外的工具调用: {response.content}"
    else:
        logger.info("模型直接返回了报告内容。将手动创建文件")
        final_report_content = response.content
        create_file.invoke({"file_name": "final_analysis_report.md", "file_contents": final_report_content})
            
    return {"final_report": final_report_content}