import json
import logging
from typing import Annotated, Literal
from langchain_core.messages import AIMessage, HumanMessage,  SystemMessage, ToolMessage
from langgraph.types import Command, interrupt
from langchain_openai import ChatOpenAI
from state import State
from prompts import *
from tools import *
os.environ["DEEPSEEK_API_KEY"] = ""
from langchain_deepseek import ChatDeepSeek
llm = ChatDeepSeek(model="deepseek-chat")
# llm = ChatOpenAI(model="qwen-plus-2025-04-28", temperature=0.0, base_url='https://dashscope.aliyuncs.com/compatible-mode/v1', api_key='')
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
    执行工具并对结果进行反思，以适应不同模型的严格要求。
    修复点：确保每次 tool_call 后都插入 ToolMessage 到消息历史中，防止 tool_call_id 遗失报错。
    """
    logger.info("***正在运行execute_node***")

    plan_object = state.get("plan", {})
    steps = plan_object.get("steps", []) if isinstance(plan_object, dict) else plan_object

    # 获取当前 pending 任务
    current_task = None
    task_index = -1
    for i, step in enumerate(steps):
        if step.get("status") == "pending":
            current_task = step
            task_index = i
            break

    # 如果没有任务，生成报告
    if current_task is None:
        logger.info("所有任务已完成，准备生成报告。")
        summary_of_past_steps = "\n\n---\n\n".join(
            [f"## {title}\n\n**结果:**\n```\n{result}\n```" for title, result in state.get("past_steps", [])]
        )
        observation_message = [
            HumanMessage(content=f"这是之前所有步骤的执行摘要和结果，请根据这些信息生成一份最终的、详细的分析报告：\n\n{summary_of_past_steps}")
        ]
        return Command(goto='report', update={"observations": observation_message})

    logger.info(f"当前执行STEP:{current_task}")

    # 构造调用模型的 prompt
    messages = list(state.get("messages", []))
    user_msg = HumanMessage(content=EXECUTION_PROMPT.format(
        user_message=state.get('user_message', ''), step=current_task))
    messages.append(user_msg)

    # 第一次调用：选择工具
    response = llm.bind_tools([create_file, str_replace, shell_exec]).invoke(messages)
    logger.info(f"=================模型返回===============\n{response}")

    # 情况 1：无工具调用，直接写入历史并结束此步
    if not response.tool_calls:
        logger.info("模型没有返回工具调用，将记录其回复并继续下一步。")
        steps[task_index]['status'] = 'completed'
        return Command(goto="execute", update={
            "plan": plan_object,
            "messages": messages + [response],
        })

    # 情况 2：执行工具
    tool_call = response.tool_calls[0]
    tool_call_id = tool_call.get("id")
    tool_name = tool_call.get("name")
    tool_args = tool_call.get("args")

    tools = {"create_file": create_file, "str_replace": str_replace, "shell_exec": shell_exec}
    if tool_name not in tools:
        raise ValueError(f"Unknown tool: {tool_name}")
    
    tool = tools[tool_name]
    tool_result = tool.invoke(tool_args)
    logger.info(f"tool_name:{tool_name}, tool_args:{tool_args}\ntool_result:{tool_result}")

    # 创建 ToolMessage 并组合新的消息序列
    tool_message = ToolMessage(content=json.dumps(tool_result, ensure_ascii=False), tool_call_id=tool_call_id)
    messages_for_reflection = messages + [response, tool_message]

    # 第二次调用：反思总结
    final_response = llm.invoke(messages_for_reflection)

    # 状态更新
    steps[task_index]['status'] = 'completed'
    past_steps = state.get('past_steps', [])
    past_steps.append((f"步骤 {task_index+1}: {current_task['title']}", str(tool_result)))

    updated_state = {
        "plan": plan_object,
        "messages": messages_for_reflection + [final_response],  # ✅ 修复点：完整保留 ToolMessage + final_response
        "past_steps": past_steps
    }

    logger.info(f"✅ 步骤 {task_index+1} 执行完成，状态已更新。")
    return Command(goto='execute', update=updated_state)

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
        else:
            final_report_content = f"报告生成期间出现意外的工具调用: {response.content}"
    else:
        logger.info("模型直接返回了报告内容。将手动创建文件")
        final_report_content = response.content
        create_file.invoke({"file_name": "final_analysis_report.md", "file_contents": final_report_content})
            
    return {"final_report": final_report_content}

