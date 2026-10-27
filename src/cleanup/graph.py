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
    graph.add_node(node="load_dataset", action=load_dataset)
    graph.add_node(node="assess_images", action=assess_images)
    graph.add_node(node="judge_borderline", action=make_judge_node(judge))
    graph.add_node(node="write_output", action=write_output)
    graph.add_node(node="write_report", action=write_report)
    graph.add_edge(start_key=START, end_key="load_dataset")
    graph.add_edge(start_key="load_dataset", end_key="assess_images")
    graph.add_conditional_edges(
        "assess_images", route_after_assess, ["judge_borderline", "write_output"]
    )
    graph.add_edge(start_key="judge_borderline", end_key="write_output")
    graph.add_edge(start_key="write_output", end_key="write_report")
    graph.add_edge(start_key="write_report", end_key=END)
    return graph.compile()
