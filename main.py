import asyncio
from pathlib import Path

from agent_framework import (
    InlineSkill,
    MCPStreamableHTTPTool,
    SkillFrontmatter,
    SkillsProvider,
    tool,
)
from agent_framework.foundry import FoundryChatClient
from agent_framework.orchestrations import SequentialBuilder
from azure.identity import AzureCliCredential



# Create chat clients using FoundryChatClient for complex and simple models
chat_client_complex = FoundryChatClient(
    project_endpoint="https://sbiryukov.services.ai.azure.com/api/projects/proj-default",
    model="gpt-5.6-sol",
    credential=AzureCliCredential(),
)

chat_client_simple = FoundryChatClient(
    project_endpoint="https://sbiryukov.services.ai.azure.com/api/projects/proj-default",
    model="gpt-4.1-mini",
    credential=AzureCliCredential(),
)

# create agent tools
@tool
def stop_words() -> list[str]:
    """Return words that must never appear in the article, such as negative or low-value terms."""
    print("Tool stop_words called")
    return ["cheap", "expensive", "low-quality", "inferior", "mediocre", "subpar", "poor", "bad", "terrible", "good"]


@tool
def must_use_words() -> list[str]:
    """Return preferred positive words that should be used in the article when appropriate."""
    print("Tool must_use_words called")
    return ["innovative", "high-quality", "durable", "reliable", "premium", "excellent", "outstanding", "superior", "exceptional", "best"]


@tool
def replace_umlauts(message: str) -> str:
    """Transliterate German umlauts and ß into ASCII digraphs (ä→ae, ö→oe, ü→ue, ß→ss)."""
    print("Tool replace_umlauts called")
    return (
        message.replace("ä", "ae")
               .replace("ö", "oe")
               .replace("ü", "ue")
               .replace("ß", "ss")
    )


# Remote MCP server exposing Microsoft Learn documentation search tools
ms_learn = MCPStreamableHTTPTool(
    name="microsoft_learn",
    url="https://learn.microsoft.com/api/mcp",
    approval_mode="never_require",
)


# Create the research -> writer -> editor -> translator pipeline agents
researcher = chat_client_complex.as_agent(
    instructions=(
        "You are a technical researcher. Given the name of a Microsoft technology, use the "
        "Microsoft Learn tools to find relevant, accurate, up-to-date information about it. "
        "Return concise factual notes (key features, use cases, and important details) that a "
        "writer can turn into a short article. Do not write the article yourself."
    ),
    tools=[ms_learn],
    name="researcher",
)

writer = chat_client_complex.as_agent(
    instructions=(
        "You are a technical writer. Using the researcher's notes, draft a short article "
        "(2-3 paragraphs) about the Microsoft technology. Call the stop_words tool and never use "
        "any of those words; call the must_use_words tool and include suitable words from it where "
        "natural. Return only the article text."
    ),
    tools=[stop_words, must_use_words],
    name="writer",
)

editor = chat_client_complex.as_agent(
    instructions=(
        "You are an editor. Review the writer's article and adjust it for a young audience: make "
        "it clear, energetic, and engaging while keeping it accurate. Return only the revised "
        "English article."
    ),
    name="editor",
)


# Skill that lets the translator persist the finished article to the filesystem.
# The skill's instructions are advertised to the agent via progressive disclosure,
# and its save_summary script runs in-process as a tool.
save_summary_skill = InlineSkill(
    frontmatter=SkillFrontmatter(
        name="save-summary",
        description="Save the final article to the local filesystem under summaries/.",
    ),
    instructions=(
        "Use this skill to persist the finished article. Run the save_summary script with two "
        "arguments: 'filename' (a short, safe name ending in .txt) and 'content' (the full final "
        "article text). The script writes the file into the summaries/ folder and returns the "
        "absolute path where it was saved."
    ),
)


@save_summary_skill.script(
    name="save_summary",
    description="Save text to summaries/<filename> and return the absolute path.",
)
def save_summary(filename: str, content: str) -> str:
    out = Path("summaries") / filename
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(content, encoding="utf-8")
    print(f"Skill save_summary wrote {out.resolve()}")
    return str(out.resolve())


# Expose the skill to an agent; disable approval so it runs unattended.
save_summary_skill_provider = SkillsProvider(
    [save_summary_skill],
    disable_load_skill_approval=True,
    disable_run_skill_script_approval=True,
)


translator = chat_client_simple.as_agent(
    instructions=(
        "You receive the editor's final article. First translate the whole article into German, "
        "preserving its meaning and tone. Then pass the complete German translation through the "
        "replace_umlauts tool so no umlauts or ß remain. Finally, use the save-summary skill to "
        "persist the transliterated article to disk: load the skill, then run its save_summary "
        "script with a short filename based on the topic (e.g. 'azure-functions.txt') and the "
        "article as content. Return ONLY the transliterated German article."
    ),
    tools=[replace_umlauts],
    context_providers=[save_summary_skill_provider],
    name="translator",
)


# Build sequential workflow and expose every participant's response as output
workflow = SequentialBuilder(
    participants=[researcher, writer, editor, translator],
    chain_only_agent_responses=True,
    output_from=[researcher, writer, editor, translator],
).build()



# Run and print every agent's response
async def main():
    # Connect to the remote MCP server for the duration of the run.
    async with ms_learn:
        events = await workflow.run("Azure Functions")
    outputs = events.get_outputs()

    if outputs:
        for i, output in enumerate(outputs):
            print(f"===== Response {i + 1} =====")
            # Map tool call ids to their tool name so results can be labeled too.
            call_names = {
                c.call_id: c.name
                for msg in output.messages
                for c in msg.contents
                if c.type == "function_call"
            }
            for msg in output.messages:
                name = msg.author_name or "assistant"
                for content in msg.contents:
                    if content.type == "text":
                        print(f"[{name}]\n{content.text}")
                    elif content.type == "function_call":
                        print(f"[{name}] (tool call: {content.name})")
                    elif content.type == "function_result":
                        tool = call_names.get(content.call_id, "unknown")
                        print(f"[{name}] (tool result: {tool})")
                    else:
                        print(f"[{name}] ({content.type})")
                    print()


if __name__ == "__main__":
    asyncio.run(main())