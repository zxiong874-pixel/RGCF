"""Recording preflight topic checks."""

from dataclasses import dataclass

from bev_obstacle_detection.baseline_plan import baseline_plan


@dataclass(frozen=True)
class TopicCheckResult:
    method: str
    expected_topics: tuple
    available_topics: tuple
    missing_topics: tuple
    ok: bool


def plans_for_methods(config, methods):
    plans = {item.method: item for item in baseline_plan(config)}
    if methods == ("all",):
        return tuple(plans[method] for method in config.methods)

    selected = []
    for method in methods:
        if method not in plans:
            raise ValueError("unknown method: {}".format(method))
        selected.append(plans[method])
    return tuple(selected)


def expected_topics_for_plan(plan, include_outputs=False):
    topics = list(plan.required_topics)
    if include_outputs:
        topics.extend(plan.output_topics)
    seen = set()
    unique_topics = []
    for topic in topics:
        if topic and topic not in seen:
            unique_topics.append(topic)
            seen.add(topic)
    return tuple(unique_topics)


def normalize_available_topics(available_topics):
    normalized = []
    for item in available_topics:
        if isinstance(item, (tuple, list)) and item:
            normalized.append(str(item[0]))
        else:
            normalized.append(str(item))
    return tuple(sorted(normalized))


def check_topics_for_plan(plan, available_topics, include_outputs=False):
    available = normalize_available_topics(available_topics)
    available_set = set(available)
    expected = expected_topics_for_plan(plan, include_outputs=include_outputs)
    missing = tuple(topic for topic in expected if topic not in available_set)
    return TopicCheckResult(
        method=plan.method,
        expected_topics=expected,
        available_topics=available,
        missing_topics=missing,
        ok=not missing,
    )


def summarize_topic_checks(results, master_error=None):
    return {
        "ok": master_error is None and all(result.ok for result in results),
        "master_error": master_error,
        "methods": [
            {
                "method": result.method,
                "ok": result.ok,
                "expected_topics": list(result.expected_topics),
                "missing_topics": list(result.missing_topics),
            }
            for result in results
        ],
    }
