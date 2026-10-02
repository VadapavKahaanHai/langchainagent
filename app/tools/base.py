"""The filing-review agent's active toolkit."""

from app.tools.filing_review import review_filing

REGISTERED_TOOLS = [review_filing]


def get_all_tools():
    return REGISTERED_TOOLS


def get_tool_definitions():
    return [
        {
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.get_input_schema().model_json_schema(),
        }
        for tool in REGISTERED_TOOLS
    ]
