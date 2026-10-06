import argparse
import csv
import json
import os
import re
from pathlib import Path
from typing import Any, Dict

import pandas as pd
from dotenv import load_dotenv

load_dotenv()

try:
    from google import genai
except ImportError:
    genai = None


BASE = Path(__file__).resolve().parent
DATA = BASE / "data"


def load_quoted_row_csv(path: Path) -> pd.DataFrame:
    raw = path.read_text(encoding="utf-8-sig").splitlines()

    if not raw:
        return pd.DataFrame()

    lines = []

    for line in raw:
        line = line.strip()

        if len(line) >= 2 and line[0] == '"' and line[-1] == '"':
            line = line[1:-1].replace('""', '"')

        lines.append(line)

    rows = list(csv.reader(lines))

    return pd.DataFrame(rows[1:], columns=rows[0])


def load_data():
    sales = load_quoted_row_csv(DATA / "sales_data.csv")
    targets = load_quoted_row_csv(DATA / "targets.csv")

    sales.columns = [
        c.strip().strip('"').lstrip("\ufeff")
        for c in sales.columns
    ]

    targets.columns = [
        c.strip().strip('"').lstrip("\ufeff")
        for c in targets.columns
    ]

    numeric_columns = [
        "order_id",
        "quantity",
        "unit_price",
        "discount",
        "shipping_cost",
        "profit"
    ]

    for column in numeric_columns:
        if column in sales.columns:
            sales[column] = pd.to_numeric(
                sales[column],
                errors="coerce"
            )

    sales["order_date"] = pd.to_datetime(
        sales["order_date"],
        errors="coerce"
    )

    sales["month"] = (
        sales["order_date"]
        .dt.to_period("M")
        .astype(str)
    )

    sales["year"] = sales["order_date"].dt.year

    sales["revenue"] = (
        sales["quantity"]
        * sales["unit_price"]
        * (1 - sales["discount"])
    )

    sales["orders"] = 1

    targets["target_revenue"] = pd.to_numeric(
        targets["target_revenue"],
        errors="coerce"
    )

    return sales, targets


def load_dictionary():
    path = DATA / "data_dictionary.json"

    try:
        return json.loads(
            path.read_text(encoding="utf-8-sig")
        )

    except Exception:

        return {
            "metrics": {
                "revenue": "quantity * unit_price * (1 - discount)",
                "profit": "profit",
                "orders": "count(order_id)",
                "avg_order_value": "revenue / orders"
            },

            "dimensions": [
                "region",
                "country",
                "city",
                "customer_segment",
                "product_category",
                "product_subcategory",
                "product_name",
                "customer_id",
                "order_date"
            ],

            "synonyms": {
                "sales": "revenue",
                "income": "revenue",
                "earnings": "profit",
                "orders": "count(order_id)",
                "aov": "avg_order_value"
            },

            "time_mappings": {
                "last month": "previous calendar month",
                "this quarter": "current quarter",
                "yoy": "year over year"
            }
        }


def load_evaluation_queries():

    path = DATA / "nl_queries.json"

    try:
        return json.loads(
            path.read_text(encoding="utf-8-sig")
        )

    except Exception:

        text = path.read_text(
            encoding="utf-8-sig"
        )

        queries = re.findall(
            r'""query"":\s*""([^"]+)""',
            text
        )

        logics = re.findall(
            r'""expected_logic"":\s*""([^"]+)"""',
            text
        )

        return [
            {
                "query": q,
                "expected_logic": logic
            }
            for q, logic in zip(queries, logics)
        ]


def load_feedback():

    path = DATA / "feedback_log.csv"

    if not path.exists():
        return []

    try:

        df = load_quoted_row_csv(path)

        return df.fillna("").to_dict(
            "records"
        )

    except Exception:

        return []


PLANNER_SCHEMA = {
    "metric": "revenue|profit|orders|avg_order_value",

    "group_by": [
        "dimension"
    ],

    "filters": [
        {
            "field": "field",
            "operator": "=|!=|>|>=|<|<=",
            "value": "value"
        }
    ],

    "aggregation": "sum|count|avg",

    "rank": {
        "n": 2,
        "order": "desc|asc",
        "within_groups": False
    },

    "contribution_percent": False,

    "target_comparison": False,

    "time_granularity": "month|year|none",

    "yoy": False,

    "nested": None
}


def gemini_plan(
    query: str,
    dictionary: dict,
    feedback: list
):

    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key or genai is None:
        return rule_based_plan(query), False

    client = genai.Client(
        api_key=api_key
    )

    model = os.getenv(
        "GEMINI_MODEL",
        "gemini-2.5-flash"
    )

    prompt = f"""
You are an intelligent analytics query planner.

Convert the user's natural language business query
into ONLY the JSON analytical DSL below.

Never generate Python code.
Never generate SQL.
Never invent database columns.

Allowed DSL:

{json.dumps(PLANNER_SCHEMA, indent=2)}

Data dictionary:

{json.dumps(dictionary, indent=2)}

Important rules:

1. "sales" means revenue.
2. "income" means revenue.
3. "earnings" means profit.
4. "orders" means count of orders.
5. "AOV" means average order value.
6. Use ISO month format YYYY-MM.
7. "top N cities" means ranking cities by the requested metric.
8. "top N per region" means within_groups=true.
9. "contribution %" means contribution_percent=true.
10. Target questions mean target_comparison=true.
11. YoY means yoy=true.
12. "top product in each region" should use:
    group_by=["region","product_name"]
    and within_groups=true.
13. "revenue of top 3 customers per region" should use nested
    customer ranking followed by region aggregation.
14. Only use fields present in the data dictionary.

Return ONLY valid JSON.

User query:

{query}

Previous feedback:

{json.dumps(feedback[:5], indent=2)}
"""

    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config={
            "response_mime_type": "application/json"
        }
    )

    plan = json.loads(response.text)

    return plan, True


def rule_based_plan(query: str):

    q = query.lower()

    plan = {
        "metric": "revenue",
        "group_by": [],
        "filters": [],
        "aggregation": "sum",
        "rank": None,
        "contribution_percent": False,
        "target_comparison": False,
        "time_granularity": "none",
        "yoy": False,
        "nested": None
    }

    if "profit" in q or "earnings" in q:
        plan["metric"] = "profit"

    elif (
        "average order" in q
        or "aov" in q
    ):
        plan["metric"] = "avg_order_value"
        plan["aggregation"] = "avg"

    elif "order" in q:
        plan["metric"] = "orders"
        plan["aggregation"] = "count"

    dimensions = [
        "region",
        "country",
        "city",
        "customer_segment",
        "product_category",
        "product_subcategory",
        "product_name",
        "customer_id"
    ]

    for dimension in dimensions:

        if dimension.replace(
            "_",
            " "
        ) in q:

            plan["group_by"].append(
                dimension
            )

    if "contribution" in q:
        plan["contribution_percent"] = True

    if (
        "target" in q
        or "missed" in q
    ):
        plan["target_comparison"] = True

    if (
        "yoy" in q
        or "year over year" in q
    ):
        plan["yoy"] = True

    match = re.search(
        r"top\s+(\d+)",
        q
    )

    if match:

        plan["rank"] = {
            "n": int(match.group(1)),
            "order": "desc",
            "within_groups": (
                "each" in q
                or "per" in q
            )
        }

    months = {
        "january": "01",
        "february": "02",
        "march": "03",
        "april": "04",
        "may": "05",
        "june": "06",
        "july": "07",
        "august": "08",
        "september": "09",
        "october": "10",
        "november": "11",
        "december": "12"
    }

    for name, number in months.items():

        if name in q:

            plan["filters"].append(
                {
                    "field": "month",
                    "operator": "=",
                    "value": f"2024-{number}"
                }
            )

    if "top product in each region" in q:

        plan["group_by"] = [
            "region",
            "product_name"
        ]

        plan["rank"] = {
            "n": 1,
            "order": "desc",
            "within_groups": True
        }

    if "top 3 customers per region" in q:

        plan["group_by"] = [
            "region",
            "customer_id"
        ]

        plan["rank"] = {
            "n": 3,
            "order": "desc",
            "within_groups": True
        }

        plan["nested"] = {
            "rank_entity": "customer_id",
            "final_group": "region"
        }

    if (
        "by category" in q
        and "contribution" in q
    ):

        plan["group_by"] = [
            "product_category"
        ]

    if "by region" in q:

        plan["group_by"] = [
            "region"
        ]

    if "by city" in q:

        plan["group_by"] = [
            "city"
        ]

    if "top 2 cities" in q:

        plan["group_by"] = [
            "city"
        ]

        plan["rank"] = {
            "n": 2,
            "order": "desc",
            "within_groups": False
        }

    if "missed its target in feb" in q:

        plan["filters"] = [
            {
                "field": "month",
                "operator": "=",
                "value": "2024-02"
            }
        ]

        plan["group_by"] = [
            "region"
        ]

        plan["target_comparison"] = True

    return plan


def validate_plan(
    plan: dict,
    sales: pd.DataFrame
):

    valid_fields = set(
        sales.columns
    )

    valid_metrics = {
        "revenue",
        "profit",
        "orders",
        "avg_order_value"
    }

    if plan.get("metric") not in valid_metrics:
        raise ValueError(
            "Unsupported metric"
        )

    for dimension in plan.get(
        "group_by",
        []
    ):

        if dimension not in valid_fields:

            raise ValueError(
                f"Unknown dimension: {dimension}"
            )

    for filter_item in plan.get(
        "filters",
        []
    ):

        if filter_item.get(
            "field"
        ) not in valid_fields:

            raise ValueError(
                f"Unknown filter field: "
                f"{filter_item.get('field')}"
            )

        if filter_item.get(
            "operator"
        ) not in {
            "=",
            "!=",
            ">",
            ">=",
            "<",
            "<="
        }:

            raise ValueError(
                "Unsupported filter operator"
            )


def apply_filters(
    df,
    filters
):

    result = df.copy()

    for item in filters:

        field = item["field"]
        operator = item["operator"]
        value = item["value"]

        column = result[field]

        if pd.api.types.is_numeric_dtype(
            column
        ):

            try:
                value = float(value)
            except Exception:
                pass

        if operator == "=":
            result = result[
                column == value
            ]

        elif operator == "!=":
            result = result[
                column != value
            ]

        elif operator == ">":
            result = result[
                column > value
            ]

        elif operator == ">=":
            result = result[
                column >= value
            ]

        elif operator == "<":
            result = result[
                column < value
            ]

        elif operator == "<=":
            result = result[
                column <= value
            ]

    return result


def aggregate(
    df,
    metric,
    group_by
):

    if metric == "avg_order_value":

        if group_by:

            result = (
                df.groupby(
                    group_by,
                    dropna=False
                )
                .agg(
                    revenue=(
                        "revenue",
                        "sum"
                    ),
                    orders=(
                        "orders",
                        "sum"
                    )
                )
                .reset_index()
            )

            result["value"] = (
                result["revenue"]
                /
                result["orders"]
                .replace(
                    0,
                    pd.NA
                )
            )

            return result.drop(
                columns=[
                    "revenue",
                    "orders"
                ]
            )

        return pd.DataFrame(
            {
                "value": [
                    df["revenue"].sum()
                    /
                    max(
                        len(df),
                        1
                    )
                ]
            }
        )

    if metric == "orders":

        if group_by:

            return (
                df.groupby(
                    group_by,
                    dropna=False
                )["orders"]
                .sum()
                .reset_index(
                    name="value"
                )
            )

        return pd.DataFrame(
            {
                "value": [
                    df["orders"].sum()
                ]
            }
        )

    if group_by:

        return (
            df.groupby(
                group_by,
                dropna=False
            )[metric]
            .sum()
            .reset_index(
                name="value"
            )
        )

    return pd.DataFrame(
        {
            "value": [
                df[metric].sum()
            ]
        }
    )


def execute(
    plan,
    sales,
    targets
):

    validate_plan(
        plan,
        sales
    )

    df = apply_filters(
        sales,
        plan.get(
            "filters",
            []
        )
    )

    metric = plan["metric"]

    group_by = plan.get(
        "group_by",
        []
    )

    # Nested ranking
    if plan.get(
        "nested"
    ):

        entity = plan[
            "nested"
        ]["rank_entity"]

        parent = plan[
            "nested"
        ]["final_group"]

        grouped = aggregate(
            df,
            metric,
            [
                parent,
                entity
            ]
        )

        grouped["rank"] = (
            grouped
            .groupby(parent)["value"]
            .rank(
                method="first",
                ascending=False
            )
        )

        grouped = grouped[
            grouped["rank"]
            <=
            plan["rank"]["n"]
        ]

        return (
            grouped
            .groupby(
                parent,
                as_index=False
            )["value"]
            .sum()
        )

    # Target comparison
    if plan.get(
        "target_comparison"
    ):

        if "region" not in group_by:

            group_by = [
                "region"
            ] + group_by

        grouped = aggregate(
            df,
            metric,
            group_by
        )

        month = (
            df["month"].max()
            if len(df)
            else None
        )

        grouped["month"] = month

        joined = grouped.merge(
            targets,
            on=[
                "region",
                "month"
            ],
            how="left"
        )

        joined[
            "missed_target"
        ] = (
            joined["value"]
            <
            joined["target_revenue"]
        )

        return joined

    # YoY
    if plan.get(
        "yoy"
    ):

        grouped = (
            df.groupby(
                "year"
            )[metric]
            .sum()
            .reset_index(
                name="value"
            )
        )

        grouped[
            "previous_year"
        ] = grouped[
            "value"
        ].shift(1)

        grouped[
            "yoy_growth_pct"
        ] = (
            (
                grouped["value"]
                -
                grouped["previous_year"]
            )
            /
            grouped[
                "previous_year"
            ].replace(
                0,
                pd.NA
            )
            * 100
        )

        return grouped

    result = aggregate(
        df,
        metric,
        group_by
    )

    # Contribution percentage
    if plan.get(
        "contribution_percent"
    ):

        total = result[
            "value"
        ].sum()

        if total:

            result[
                "contribution_pct"
            ] = (
                result["value"]
                /
                total
                * 100
            )

        else:

            result[
                "contribution_pct"
            ] = 0

    # Ranking
    rank = plan.get(
        "rank"
    )

    if rank:

        if (
            rank.get(
                "within_groups"
            )
            and len(group_by) >= 2
        ):

            parent = group_by[:-1]

            result = result.sort_values(
                parent + ["value"],
                ascending=[
                    True
                ] * len(parent)
                + [
                    False
                ]
            )

            result["rank"] = (
                result
                .groupby(parent)["value"]
                .rank(
                    method="first",
                    ascending=False
                )
            )

            result = result[
                result["rank"]
                <= rank["n"]
            ]

        else:

            result = (
                result
                .sort_values(
                    "value",
                    ascending=False
                )
                .head(
                    rank["n"]
                )
            )

    return result


def confidence(
    plan,
    result,
    used_genai
):

    score = (
        0.80
        if used_genai
        else 0.55
    )

    if plan.get("group_by"):
        score += 0.05

    if plan.get("filters"):
        score += 0.05

    if plan.get("rank"):
        score += 0.05

    if (
        plan.get("target_comparison")
        or plan.get("yoy")
        or plan.get("contribution_percent")
    ):

        score += 0.05

    if (
        isinstance(
            result,
            pd.DataFrame
        )
        and len(result) == 0
    ):

        score -= 0.10

    return round(
        max(
            0,
            min(
                1,
                score
            )
        ),
        2
    )


def answer(
    query: str
):

    sales, targets = load_data()

    dictionary = load_dictionary()

    feedback = load_feedback()

    try:

        plan, used_genai = gemini_plan(
            query,
            dictionary,
            feedback
        )

        result = execute(
            plan,
            sales,
            targets
        )

        score = confidence(
            plan,
            result,
            used_genai
        )

        if used_genai:

            explanation = (
                f"Gemini interpreted the query as "
                f"metric={plan['metric']}, "
                f"group_by={plan.get('group_by', [])}, "
                f"filters={plan.get('filters', [])}, "
                f"rank={plan.get('rank')}. "
                f"The generated analytical plan was validated "
                f"and executed deterministically with pandas."
            )

        else:

            explanation = (
                f"Understood the query as "
                f"metric={plan['metric']}, "
                f"group_by={plan.get('group_by', [])}, "
                f"filters={plan.get('filters', [])}, "
                f"rank={plan.get('rank')}. "
                f"Gemini was unavailable, so the local "
                f"fallback planner was used."
            )

        serial = (
            result.to_dict(
                orient="records"
            )
            if isinstance(
                result,
                pd.DataFrame
            )
            else result
        )

        return {
            "query": query,

            "generated_logic": json.dumps(
                plan,
                separators=(
                    ",",
                    ":"
                )
            ),

            "result": json.dumps(
                serial,
                default=str
            ),

            "confidence_score": score,

            "explanation": explanation
        }

    except Exception as error:

        return {
            "query": query,
            "generated_logic": "",
            "result": "",
            "confidence_score": 0.0,
            "explanation": (
                "Query could not be executed safely: "
                f"{type(error).__name__}: {error}"
            )
        }


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--query"
    )

    parser.add_argument(
        "--batch"
    )

    args = parser.parse_args()

    if args.query:

        print(
            json.dumps(
                answer(
                    args.query
                ),
                indent=2
            )
        )

        return

    if args.batch:

        items = load_evaluation_queries()

        outputs = [
            answer(
                item["query"]
            )
            for item in items
        ]

        print(
            json.dumps(
                outputs,
                indent=2
            )
        )

        return

    print(
        json.dumps(
            answer(
                "Top 2 cities by profit"
            ),
            indent=2
        )
    )


if __name__ == "__main__":
    main()