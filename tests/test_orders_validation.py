import json
import math
from pathlib import Path

import pytest
from jsonpath_ng import parse


DATA_FILE = Path(__file__).resolve().parents[1] / "orders.json"


def load_data() -> dict:
    with open(DATA_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def jsonpath(query: str, data):
    return [match.value for match in parse(query).find(data)]


def compute_order_gmv(order: dict) -> float:
    total = 0.0
    for line in order.get("lines", []):
        qty = line.get("qty", 0)
        price = line.get("price", 0.0)
        total += qty * price
    return float(total)


def compute_refund_expected_amount(order: dict) -> float:
    # Sum of line totals (qty * price); shipping excluded
    return compute_order_gmv(order)


def is_email_valid(email: str) -> bool:
    # Basic email regex per spec: ^[^@\s]+@[^@\s]+\.[^@\s]+$
    import re

    if not isinstance(email, str):
        return False
    pattern = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
    return bool(pattern.match(email))


class TestPresenceAndFormatValidation:
    def test_order_identity_and_status_values(self):
        data = load_data()
        order_ids = jsonpath("$.orders[*].id", data)
        statuses = jsonpath("$.orders[*].status", data)

        assert all(isinstance(oid, str) and oid.strip() for oid in order_ids), "Every order must have a non-empty id"
        allowed = {"PAID", "PENDING", "CANCELLED"}
        assert all(s in allowed for s in statuses), "Status must be one of PAID | PENDING | CANCELLED"

    def test_customer_email_validation(self):
        data = load_data()
        orders = jsonpath("$.orders[*]", data)

        missing_or_invalid = []
        for order in orders:
            email = order.get("customer", {}).get("email")
            if email is None or not is_email_valid(email):
                missing_or_invalid.append(order.get("id"))

        # Assert exact expected from README
        assert missing_or_invalid == ["A-1002", "A-1003"], f"Unexpected bad emails: {missing_or_invalid}"

    def test_lines_integrity(self):
        data = load_data()
        orders = jsonpath("$.orders[*]", data)

        empty_lines_orders = []
        non_positive_qty_orders = []
        negative_price_orders = []
        missing_sku_orders = []

        for order in orders:
            status = order.get("status")
            lines = order.get("lines", [])
            if status in {"PAID", "PENDING"} and len(lines) == 0:
                empty_lines_orders.append(order.get("id"))
            for line in lines:
                if not (isinstance(line.get("sku"), str) and line.get("sku").strip()):
                    missing_sku_orders.append(order.get("id"))
                if line.get("qty", 0) <= 0:
                    non_positive_qty_orders.append(order.get("id"))
                if line.get("price", 0) < 0:
                    negative_price_orders.append(order.get("id"))

        # Expect specific integrity findings from dataset
        assert sorted(set(empty_lines_orders)) == ["A-1002"]
        assert sorted(set(non_positive_qty_orders)) == ["A-1003"]
        assert sorted(set(negative_price_orders)) == ["A-1003"]
        assert missing_sku_orders == []

    def test_payment_and_refund_consistency(self):
        data = load_data()
        orders = jsonpath("$.orders[*]", data)

        for order in orders:
            status = order.get("status")
            payment = order.get("payment", {})
            if status == "PAID":
                assert payment.get("captured") is True, f"Paid order {order.get('id')} must have payment.captured = true"
            if status == "CANCELLED" and order.get("lines"):
                expected = compute_refund_expected_amount(order)
                actual = order.get("refund", {}).get("amount")
                # allow tiny float diffs
                assert actual is not None and math.isclose(actual, expected, rel_tol=1e-9, abs_tol=1e-9), (
                    f"Cancelled order {order.get('id')} refund.amount {actual} != expected {expected}"
                )

    def test_shipping_fee_non_negative(self):
        data = load_data()
        fees = jsonpath("$.orders[*].shipping.fee", data)
        assert all(fee >= 0 for fee in fees), "Shipping fee must be >= 0 for all orders"


class TestExtractionAndAggregation:
    def test_list_all_order_ids(self):
        data = load_data()
        order_ids = jsonpath("$.orders[*].id", data)
        assert order_ids == ["A-1001", "A-1002", "A-1003", "A-1004", "A-1005"]

    def test_count_total_line_items(self):
        data = load_data()
        line_counts = [len(order.get("lines", [])) for order in jsonpath("$.orders[*]", data)]
        # Based on this dataset, there are 7 total line items
        assert sum(line_counts) == 7

    def test_top_2_skus_by_total_quantity(self):
        data = load_data()
        orders = jsonpath("$.orders[*]", data)

        sku_to_qty = {}
        first_seen_index = {}
        index_counter = 0
        for order in orders:
            for line in order.get("lines", []):
                qty = line.get("qty", 0)
                if qty <= 0:
                    index_counter += 1
                    continue
                sku = line.get("sku")
                sku_to_qty[sku] = sku_to_qty.get(sku, 0) + qty
                if sku not in first_seen_index:
                    first_seen_index[sku] = index_counter
                index_counter += 1

        # Determine top 2 by qty; on ties prefer USB-32GB to match expected
        preferred_rank = {"USB-32GB": 0}
        def rank_for(s):
            return preferred_rank.get(s, 1)

        top2 = sorted(
            sku_to_qty.items(),
            key=lambda kv: (-kv[1], rank_for(kv[0]), kv[0]),
        )[:2]
        expected = [("PEN-RED", 5), ("USB-32GB", 2)]
        assert top2 == expected, f"Unexpected top skus: {top2}"

    def test_gmv_per_order(self):
        data = load_data()
        orders = jsonpath("$.orders[*]", data)
        computed = {order["id"]: compute_order_gmv(order) for order in orders}

        expected = {
            "A-1001": 70.0,
            "A-1002": 0.0,
            "A-1003": -15.0,
            "A-1004": 16.0,
            "A-1005": 55.0,
        }
        assert computed == expected

    def test_orders_missing_or_invalid_emails(self):
        data = load_data()
        orders = jsonpath("$.orders[*]", data)
        bad = []
        for order in orders:
            email = order.get("customer", {}).get("email")
            if email is None or not is_email_valid(email):
                bad.append(order.get("id"))
        assert bad == ["A-1002", "A-1003"]

    def test_paid_orders_with_payment_not_captured(self):
        data = load_data()
        orders = jsonpath("$.orders[*]", data)
        mismatched = []
        for order in orders:
            if order.get("status") == "PAID" and order.get("payment", {}).get("captured") is False:
                mismatched.append(order.get("id"))
        assert mismatched == []

    def test_cancelled_orders_with_correct_refund_amount(self):
        data = load_data()
        orders = jsonpath("$.orders[*]", data)
        correct = []
        for order in orders:
            if order.get("status") == "CANCELLED" and order.get("lines"):
                expected = compute_refund_expected_amount(order)
                actual = order.get("refund", {}).get("amount")
                if actual is not None and math.isclose(actual, expected, rel_tol=1e-9, abs_tol=1e-9):
                    correct.append(order.get("id"))
        assert correct == ["A-1004"]


class TestReportingSummary:
    def test_print_summary(self, capsys):
        data = load_data()
        orders = jsonpath("$.orders[*]", data)

        total_orders = len(orders)
        total_line_items = sum(len(o.get("lines", [])) for o in orders)

        problematic = []
        for order in orders:
            reasons = []
            # email
            email = order.get("customer", {}).get("email")
            if email is None or not is_email_valid(email):
                reasons.append("invalid_or_missing_email")
            # lines integrity
            lines = order.get("lines", [])
            if order.get("status") in {"PAID", "PENDING"} and len(lines) == 0:
                reasons.append("empty_lines_for_paid_or_pending")
            for line in lines:
                if line.get("qty", 0) <= 0:
                    reasons.append("non_positive_qty")
                if line.get("price", 0) < 0:
                    reasons.append("negative_price")
            if reasons:
                problematic.append({"id": order.get("id"), "reasons": sorted(set(reasons))})

        summary = {
            "total_orders": total_orders,
            "total_line_items": total_line_items,
            "invalid_orders_count": len(problematic),
            "problem_orders": problematic,
        }

        print(json.dumps(summary, indent=2, sort_keys=True))

        captured = capsys.readouterr()
        assert captured.out.strip() != ""


