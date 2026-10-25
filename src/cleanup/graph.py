"""Graph wiring for the cleanup pipeline."""

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from cleanup.judge import Judge
from cleanup.nodes import (
    assess_images,
    load_dataset,
    make_judge_node,
    route_after_assess,
    write_output,
    write_report,
)
from cleanup.state import CleanupState


def build_graph(judge: Judge | None = None) -> CompiledStateGraph:
    """Build and compile the cleanup graph.

    load_dataset -> assess_images -> [judge_borderline] -> write_output -> write_report
    """
    graph = StateGraph(CleanupState)
    graph.add_node("load_dataset", load_dataset)
    graph.add_node("assess_images", assess_images)
    graph.add_node("judge_borderline", make_judge_node(judge))
    graph.add_node("write_output", write_output)
    graph.add_node("write_report", write_report)

    graph.add_edge(START, "load_dataset")
    graph.add_edge("load_dataset", "assess_images")
    graph.add_conditional_edges(
        "assess_images", route_after_assess, ["judge_borderline", "write_output"]
    )
    graph.add_edge("judge_borderline", "write_output")
    graph.add_edge("write_output", "write_report")
    graph.add_edge("write_report", END)
    return graph.compile()
