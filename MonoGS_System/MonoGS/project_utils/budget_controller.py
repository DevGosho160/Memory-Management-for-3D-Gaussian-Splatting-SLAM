"""Shared strict live-row admission planner for all retention rankings."""

from dataclasses import dataclass
import numpy as np


class BudgetInfeasible(RuntimeError):
    pass


@dataclass
class RowPlan:
    keep: np.ndarray
    attempted_growth: int
    admitted_growth: int
    rejected_growth: int
    target_existing: int
    protected_count: int


class RowBudgetController:
    def __init__(self, maximum, low_watermark_fraction=0.95):
        if maximum < 1 or not 0 < low_watermark_fraction <= 1:
            raise ValueError("Invalid row ceiling or watermark")
        self.maximum = int(maximum)
        self.low = int(np.floor(maximum * low_watermark_fraction))
        self.max_observed = 0
        self.violations = 0
        self.attempted_growth = 0
        self.admitted_growth = 0
        self.rejected_growth = 0

    def observe(self, count):
        self.max_observed = max(self.max_observed, count)
        if count > self.maximum:
            self.violations += 1
            raise RuntimeError(f"Gaussian row ceiling violated: {count}>{self.maximum}")

    def plan(self, count, gross_growth, priority, protected, atomic_costs=None):
        """Plan before allocation. atomic_costs allow indivisible two-child splits."""
        self.observe(count)
        protected = np.asarray(protected, dtype=bool)
        priority = np.asarray(priority, dtype=np.int64)
        if protected.size != count or priority.size != count:
            raise ValueError("Ranking/protection length mismatch")
        if len(np.unique(priority)) != count or np.any((priority < 0) | (priority >= count)):
            raise ValueError("Ranking must be a permutation")
        costs = list(atomic_costs) if atomic_costs is not None else [1] * gross_growth
        if any(cost < 1 for cost in costs) or sum(costs) != gross_growth:
            raise ValueError("Invalid atomic growth costs")
        protected_count = int(protected.sum())
        if protected_count > self.maximum or (count and self.maximum < 1):
            raise BudgetInfeasible("Protected support cannot fit row ceiling")
        # Reclaim to low watermark only when planned staging would exceed K.
        pressure = count + gross_growth > self.maximum
        floor = max(protected_count, int(count > 0))
        effective_low = max(self.low, floor)
        capacity = (effective_low - floor) if pressure else (self.maximum - count)
        admitted = 0
        for cost in costs:
            if admitted + cost <= capacity:
                admitted += cost
            else:
                break
        target = min(count, effective_low - admitted) if pressure else count
        if target < floor:
            raise BudgetInfeasible("Protected support leaves no admission headroom")
        if target == count:
            keep = np.ones(count, dtype=bool)
        else:
            keep = protected.copy()
            need = target - protected_count
            if need:
                unprotected = priority[~protected[priority]]
                keep[unprotected[:need]] = True
        kept_count = int(keep.sum())
        if kept_count != target:
            raise BudgetInfeasible("Cannot meet retention target")
        if kept_count + admitted > self.maximum:
            raise AssertionError("Gross staged addition exceeds row ceiling")
        rejected = gross_growth - admitted
        self.attempted_growth += gross_growth
        self.admitted_growth += admitted
        self.rejected_growth += rejected
        return RowPlan(keep, gross_growth, admitted, rejected, target,
                       protected_count)
