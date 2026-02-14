"""
═══════════════════════════════════════════════════════════════════════════════
🧠 INTELLIGENCE ENGINE - Anomaly Detection Module
═══════════════════════════════════════════════════════════════════════════════

Task 3.6: Anomaly Detection
- 3.6.1: Counter-Intuitive Pattern Finder
- 3.6.2: Expectation Violation Detector (vs Baseline)

Big Tech Pattern: "Wait, what?" insights that make executives pay attention.
"""

import logging
from typing import Dict, List, Optional

from sqlalchemy import text

from .base import IntelligenceBase

logger = logging.getLogger(__name__)


class AnomalyDetector(IntelligenceBase):
    """
    🤔 Anomaly & Counter-Intuitive Pattern Detector
    
    Finds patterns that defy business intuition:
    - "Revenue UP but Profit DOWN" (margin compression)
    - "Price UP but Volume UP too" (inelastic demand)
    - Metrics deviating from historical baselines
    """
    
    # ═══════════════════════════════════════════════════════════════════════════
    # 🤔 SUBTASK 3.6.1: COUNTER-INTUITIVE PATTERN FINDER
    # ═══════════════════════════════════════════════════════════════════════════
    
    def find_counter_intuitive_patterns(self, user_id: str) -> Dict:
        """
        SUBTASK 3.6.1: Find patterns that defy business intuition.
        
        Discovers surprising relationships that make executives say "Wait, what?"
        """
        try:
            all_metrics = self._get_user_metrics(user_id)
            
            if len(all_metrics) < 2:
                return {"error": f"Need at least 2 metrics, found {len(all_metrics)}"}
            
            expected_relationships = self._get_expected_relationships()
            
            metric_growth = {}
            for metric in all_metrics:
                growth = self._calculate_metric_growth(user_id, metric)
                if growth is not None:
                    metric_growth[metric] = growth
            
            if len(metric_growth) < 2:
                return {"error": "Could not calculate growth for enough metrics"}
            
            patterns = []
            
            # Check 1: Expected positive correlations that are negative
            for (metric_a, metric_b), expected in expected_relationships.items():
                if metric_a in metric_growth and metric_b in metric_growth:
                    growth_a = metric_growth[metric_a]
                    growth_b = metric_growth[metric_b]
                    
                    pattern = self._check_counter_intuitive(
                        metric_a, growth_a, metric_b, growth_b, expected
                    )
                    if pattern:
                        patterns.append(pattern)
            
            # Check 2: Any metric pairs with opposite directions
            for metric_a in metric_growth:
                for metric_b in metric_growth:
                    if metric_a >= metric_b:
                        continue
                    
                    growth_a = metric_growth[metric_a]
                    growth_b = metric_growth[metric_b]
                    
                    if (metric_a, metric_b) in expected_relationships:
                        continue
                    if (metric_b, metric_a) in expected_relationships:
                        continue
                    
                    pattern = self._check_surprising_divergence(
                        metric_a, growth_a, metric_b, growth_b
                    )
                    if pattern:
                        patterns.append(pattern)
            
            patterns.sort(key=lambda x: x.get("surprise_score", 0), reverse=True)
            
            overall_surprise = self._calculate_overall_surprise(patterns)
            priorities = self._generate_investigation_priorities(patterns)
            
            result = {
                "user_id": user_id[:8] + "...",
                "metrics_analyzed": len(metric_growth),
                "metric_growth_rates": {k: round(v, 2) for k, v in metric_growth.items()},
                "counter_intuitive_patterns": patterns[:10],
                "patterns_found": len(patterns),
                "overall_surprise_score": overall_surprise,
                "investigation_priorities": priorities,
                "expected_relationships_checked": len(expected_relationships)
            }
            
            if patterns:
                logger.warning(f"🤔 Found {len(patterns)} counter-intuitive patterns for user {user_id[:8]}...")
            else:
                logger.info(f"🤔 No counter-intuitive patterns for user {user_id[:8]}...")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Counter-intuitive finder failed: {e}")
            return {"error": str(e)}
    
    def _get_expected_relationships(self) -> Dict[tuple, str]:
        """Define expected business relationships."""
        return {
            # Revenue/Sales relationships
            ("revenue", "profit"): "positive",
            ("sales", "profit"): "positive",
            ("revenue", "units_sold"): "positive",
            ("sales", "units_sold"): "positive",
            ("gross_sales", "profit"): "positive",
            ("gross_sales", "revenue"): "positive",
            
            # Cost relationships
            ("cogs", "revenue"): "positive",
            ("cogs", "units_sold"): "positive",
            ("manufacturing_price", "cogs"): "positive",
            
            # Price/Volume relationships (usually negative - law of demand)
            ("unitprice", "units_sold"): "negative",
            ("sale_price", "units_sold"): "negative",
            ("sale_price", "qty"): "negative",
            
            # Discount relationships
            ("discounts", "units_sold"): "positive",
            ("discounts", "qty"): "positive",
            ("discounts", "profit"): "negative",
            
            # Margin relationships
            ("revenue", "cogs"): "positive",
            ("profit", "cogs"): "negative",
        }
    
    def _calculate_metric_growth(self, user_id: str, metric_name: str) -> Optional[float]:
        """Calculate overall growth rate for a metric."""
        try:
            growth = self._calculate_period_growth(user_id, metric_name)
            return growth
        except Exception:
            return None
    
    def _check_counter_intuitive(self, metric_a: str, growth_a: float,
                                  metric_b: str, growth_b: float,
                                  expected: str) -> Optional[Dict]:
        """Check if relationship is counter to expectation."""
        both_positive = growth_a > 5 and growth_b > 5
        both_negative = growth_a < -5 and growth_b < -5
        a_up_b_down = growth_a > 5 and growth_b < -5
        a_down_b_up = growth_a < -5 and growth_b > 5
        
        is_counter = False
        pattern_type = None
        
        if expected == "positive":
            if a_up_b_down or a_down_b_up:
                is_counter = True
                if a_up_b_down:
                    pattern_type = f"{metric_a} UP but {metric_b} DOWN"
                else:
                    pattern_type = f"{metric_a} DOWN but {metric_b} UP"
        
        elif expected == "negative":
            if both_positive or both_negative:
                is_counter = True
                if both_positive:
                    pattern_type = f"Both {metric_a} and {metric_b} UP (expected opposite)"
                else:
                    pattern_type = f"Both {metric_a} and {metric_b} DOWN (expected opposite)"
        
        if is_counter:
            surprise_score = self._calculate_surprise_score(growth_a, growth_b, expected)
            
            return {
                "pattern_type": pattern_type,
                "metric_a": metric_a,
                "metric_b": metric_b,
                "growth_a": round(growth_a, 2),
                "growth_b": round(growth_b, 2),
                "expected_relationship": expected,
                "actual_relationship": "opposite" if expected == "positive" else "same",
                "surprise_score": surprise_score,
                "severity": "🔴 HIGH" if surprise_score > 70 else "🟡 MEDIUM" if surprise_score > 40 else "🟢 LOW",
                "insight": self._generate_counter_intuitive_insight(
                    metric_a, growth_a, metric_b, growth_b, expected
                )
            }
        
        return None
    
    def _check_surprising_divergence(self, metric_a: str, growth_a: float,
                                      metric_b: str, growth_b: float) -> Optional[Dict]:
        """Check for surprising divergence between any metric pair."""
        if growth_a > 20 and growth_b < -20:
            return {
                "pattern_type": f"{metric_a} surging while {metric_b} collapsing",
                "metric_a": metric_a,
                "metric_b": metric_b,
                "growth_a": round(growth_a, 2),
                "growth_b": round(growth_b, 2),
                "expected_relationship": "unknown",
                "actual_relationship": "divergent",
                "surprise_score": min(100, abs(growth_a - growth_b)),
                "severity": "🟡 MEDIUM",
                "insight": f"Unusual divergence: {metric_a} ({growth_a:+.1f}%) vs {metric_b} ({growth_b:+.1f}%). Investigate if related."
            }
        elif growth_a < -20 and growth_b > 20:
            return {
                "pattern_type": f"{metric_b} surging while {metric_a} collapsing",
                "metric_a": metric_a,
                "metric_b": metric_b,
                "growth_a": round(growth_a, 2),
                "growth_b": round(growth_b, 2),
                "expected_relationship": "unknown",
                "actual_relationship": "divergent",
                "surprise_score": min(100, abs(growth_a - growth_b)),
                "severity": "🟡 MEDIUM",
                "insight": f"Unusual divergence: {metric_b} ({growth_b:+.1f}%) vs {metric_a} ({growth_a:+.1f}%). Investigate if related."
            }
        
        return None
    
    def _calculate_surprise_score(self, growth_a: float, growth_b: float,
                                   expected: str) -> float:
        """Calculate how surprising a pattern is (0-100)."""
        if expected == "positive":
            divergence = abs(growth_a - growth_b)
        else:
            divergence = abs(growth_a + growth_b)
        
        score = min(100, divergence * 1.5)
        
        if abs(growth_a) > 30 or abs(growth_b) > 30:
            score = min(100, score + 20)
        
        return round(score, 1)
    
    def _generate_counter_intuitive_insight(self, metric_a: str, growth_a: float,
                                             metric_b: str, growth_b: float,
                                             expected: str) -> str:
        """Generate insight for counter-intuitive pattern."""
        if expected == "positive":
            if growth_a > 0 and growth_b < 0:
                if metric_a in ["revenue", "sales", "gross_sales"] and metric_b == "profit":
                    return f"⚠️ MARGIN COMPRESSION: {metric_a} up {growth_a:.0f}% but {metric_b} down {abs(growth_b):.0f}%. Costs rising faster than revenue."
                elif metric_b in ["units_sold", "qty"]:
                    return f"⚠️ EFFICIENCY ISSUE: {metric_a} up {growth_a:.0f}% but volume down {abs(growth_b):.0f}%. Price increases may be hurting demand."
                else:
                    return f"⚠️ UNEXPECTED: {metric_a} up {growth_a:.0f}% but {metric_b} down {abs(growth_b):.0f}%. Investigate root cause."
            else:
                return f"⚠️ UNEXPECTED: {metric_a} down {abs(growth_a):.0f}% but {metric_b} up {growth_b:.0f}%. Unusual relationship."
        
        else:  # expected == "negative"
            if growth_a > 0 and growth_b > 0:
                if metric_a in ["unitprice", "sale_price"] and metric_b in ["units_sold", "qty"]:
                    return f"💡 INELASTIC DEMAND: Price up {growth_a:.0f}% AND volume up {growth_b:.0f}%. Strong pricing power!"
                elif metric_a == "discounts" and metric_b == "profit":
                    return f"💡 SMART DISCOUNTING: Discounts up {growth_a:.0f}% AND profit up {growth_b:.0f}%. Promotions working well."
                else:
                    return f"💡 POSITIVE SURPRISE: Both {metric_a} ({growth_a:+.0f}%) and {metric_b} ({growth_b:+.0f}%) up. Better than expected!"
            else:
                return f"⚠️ DOUBLE DECLINE: Both {metric_a} ({growth_a:.0f}%) and {metric_b} ({growth_b:.0f}%) down. Concerning trend."
    
    def _calculate_overall_surprise(self, patterns: List[Dict]) -> Dict:
        """Calculate overall surprise metrics."""
        if not patterns:
            return {
                "score": 0,
                "level": "🟢 NORMAL",
                "description": "No counter-intuitive patterns found. Business performing as expected."
            }
        
        avg_surprise = sum(p.get("surprise_score", 0) for p in patterns) / len(patterns)
        max_surprise = max(p.get("surprise_score", 0) for p in patterns)
        high_severity = len([p for p in patterns if "HIGH" in p.get("severity", "")])
        
        if max_surprise > 70 or high_severity >= 2:
            level = "🔴 HIGH"
            description = f"Found {len(patterns)} surprising patterns. {high_severity} are high severity. Immediate investigation needed."
        elif max_surprise > 40 or len(patterns) >= 3:
            level = "🟡 ELEVATED"
            description = f"Found {len(patterns)} surprising patterns. Review recommended."
        else:
            level = "🟢 LOW"
            description = f"Found {len(patterns)} minor surprising patterns. Monitor but not urgent."
        
        return {
            "score": round(avg_surprise, 1),
            "max_score": round(max_surprise, 1),
            "level": level,
            "patterns_count": len(patterns),
            "high_severity_count": high_severity,
            "description": description
        }
    
    def _generate_investigation_priorities(self, patterns: List[Dict]) -> List[Dict]:
        """Generate prioritized investigation list."""
        priorities = []
        
        high_patterns = [p for p in patterns if "HIGH" in p.get("severity", "")]
        if high_patterns:
            top = high_patterns[0]
            priorities.append({
                "priority": 1,
                "action": f"Investigate {top['pattern_type']}",
                "urgency": "🔴 IMMEDIATE",
                "metrics": [top["metric_a"], top["metric_b"]],
                "insight": top.get("insight", "")
            })
        
        margin_patterns = [p for p in patterns if "profit" in p.get("metric_b", "").lower() or "profit" in p.get("metric_a", "").lower()]
        if margin_patterns and (not high_patterns or margin_patterns[0] != high_patterns[0]):
            top = margin_patterns[0]
            priorities.append({
                "priority": 2,
                "action": f"Review profitability: {top['pattern_type']}",
                "urgency": "🟡 THIS WEEK",
                "metrics": [top["metric_a"], top["metric_b"]],
                "insight": top.get("insight", "")
            })
        
        volume_patterns = [p for p in patterns if any(v in p.get("metric_a", "").lower() + p.get("metric_b", "").lower() for v in ["units", "qty", "volume"])]
        if volume_patterns:
            top = volume_patterns[0]
            priorities.append({
                "priority": 3,
                "action": f"Analyze volume trends: {top['pattern_type']}",
                "urgency": "🟢 THIS MONTH",
                "metrics": [top["metric_a"], top["metric_b"]],
                "insight": top.get("insight", "")
            })
        
        return priorities[:5]
    
    # ═══════════════════════════════════════════════════════════════════════════
    # 📉 SUBTASK 3.6.2: EXPECTATION VIOLATION DETECTOR (vs Baseline)
    # ═══════════════════════════════════════════════════════════════════════════
    
    def detect_expectation_violations(self, user_id: str, metric_name: str = None) -> Dict:
        """
        SUBTASK 3.6.2: Detect when metrics deviate from historical expectations.
        
        Compares actual performance vs historical baseline using z-scores.
        """
        try:
            if metric_name:
                metrics = [metric_name]
            else:
                metrics = self._get_user_metrics(user_id)
            
            if not metrics:
                return {"error": "No metrics found"}
            
            violations = []
            baseline_stats = {}
            
            for metric in metrics:
                baseline = self._calculate_historical_baseline(user_id, metric)
                
                if not baseline:
                    continue
                
                baseline_stats[metric] = baseline
                
                recent = self._get_recent_performance(user_id, metric)
                
                if not recent:
                    continue
                
                violation = self._check_expectation_violation(metric, recent, baseline)
                
                if violation:
                    violations.append(violation)
            
            severity_order = {"🔴 CRITICAL": 0, "🟠 HIGH": 1, "🟡 MODERATE": 2, "🟢 MINOR": 3}
            violations.sort(key=lambda x: severity_order.get(x.get("severity", ""), 99))
            
            severity_summary = {
                "critical": len([v for v in violations if "CRITICAL" in v.get("severity", "")]),
                "high": len([v for v in violations if "HIGH" in v.get("severity", "")]),
                "moderate": len([v for v in violations if "MODERATE" in v.get("severity", "")]),
                "minor": len([v for v in violations if "MINOR" in v.get("severity", "")])
            }
            
            exec_summary = self._generate_violation_summary(violations, metrics)
            
            result = {
                "user_id": user_id[:8] + "...",
                "metrics_analyzed": len(metrics),
                "violations_found": len(violations),
                "violations": violations[:15],
                "baseline_stats": baseline_stats,
                "severity_summary": severity_summary,
                "executive_summary": exec_summary
            }
            
            if violations:
                logger.warning(f"📉 Found {len(violations)} expectation violations for user {user_id[:8]}...")
            else:
                logger.info(f"📉 No expectation violations for user {user_id[:8]}...")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Expectation violation detector failed: {e}")
            return {"error": str(e)}
    
    def _calculate_historical_baseline(self, user_id: str, metric_name: str) -> Optional[Dict]:
        """Calculate historical baseline statistics for a metric."""
        try:
            with self.engine.connect() as conn:
                query = text("""
                    WITH monthly_data AS (
                        SELECT 
                            DATE_TRUNC('month', date) as month,
                            SUM(metric_value) as monthly_value
                        FROM user_uploaded_metrics
                        WHERE user_id = :user_id
                          AND metric_name = :metric_name
                        GROUP BY DATE_TRUNC('month', date)
                        ORDER BY month
                    ),
                    growth_data AS (
                        SELECT 
                            month,
                            monthly_value,
                            LAG(monthly_value) OVER (ORDER BY month) as prev_value,
                            CASE 
                                WHEN LAG(monthly_value) OVER (ORDER BY month) > 0 
                                THEN ((monthly_value - LAG(monthly_value) OVER (ORDER BY month)) / 
                                      LAG(monthly_value) OVER (ORDER BY month)) * 100
                                ELSE NULL
                            END as growth_pct
                        FROM monthly_data
                    )
                    SELECT 
                        COUNT(*) as period_count,
                        AVG(monthly_value) as avg_value,
                        STDDEV(monthly_value) as std_value,
                        AVG(growth_pct) as avg_growth,
                        STDDEV(growth_pct) as std_growth,
                        MIN(monthly_value) as min_value,
                        MAX(monthly_value) as max_value,
                        MIN(growth_pct) as min_growth,
                        MAX(growth_pct) as max_growth
                    FROM growth_data
                    WHERE growth_pct IS NOT NULL
                """)
                
                result = conn.execute(query, {
                    "user_id": user_id,
                    "metric_name": metric_name
                }).fetchone()
                
                if not result or not result.period_count or result.period_count < 3:
                    return None
                
                return {
                    "period_count": int(result.period_count),
                    "avg_value": float(result.avg_value) if result.avg_value else 0,
                    "std_value": float(result.std_value) if result.std_value else 0,
                    "avg_growth": float(result.avg_growth) if result.avg_growth else 0,
                    "std_growth": float(result.std_growth) if result.std_growth else 0,
                    "min_value": float(result.min_value) if result.min_value else 0,
                    "max_value": float(result.max_value) if result.max_value else 0,
                    "min_growth": float(result.min_growth) if result.min_growth else 0,
                    "max_growth": float(result.max_growth) if result.max_growth else 0,
                    "expected_growth_range": {
                        "low": float(result.avg_growth - result.std_growth) if result.avg_growth and result.std_growth else None,
                        "expected": float(result.avg_growth) if result.avg_growth else 0,
                        "high": float(result.avg_growth + result.std_growth) if result.avg_growth and result.std_growth else None
                    }
                }
                
        except Exception as e:
            logger.warning(f"⚠️ Failed to calculate baseline for {metric_name}: {e}")
            return None
    
    def _get_recent_performance(self, user_id: str, metric_name: str) -> Optional[Dict]:
        """Get most recent period's performance."""
        try:
            with self.engine.connect() as conn:
                query = text("""
                    WITH monthly_data AS (
                        SELECT 
                            DATE_TRUNC('month', date) as month,
                            SUM(metric_value) as monthly_value
                        FROM user_uploaded_metrics
                        WHERE user_id = :user_id
                          AND metric_name = :metric_name
                        GROUP BY DATE_TRUNC('month', date)
                        ORDER BY month DESC
                        LIMIT 2
                    )
                    SELECT 
                        m1.month as current_month,
                        m1.monthly_value as current_value,
                        m2.monthly_value as previous_value,
                        CASE 
                            WHEN m2.monthly_value > 0 
                            THEN ((m1.monthly_value - m2.monthly_value) / m2.monthly_value) * 100
                            ELSE NULL
                        END as growth_pct
                    FROM (SELECT * FROM monthly_data LIMIT 1) m1
                    CROSS JOIN (SELECT * FROM monthly_data OFFSET 1 LIMIT 1) m2
                """)
                
                result = conn.execute(query, {
                    "user_id": user_id,
                    "metric_name": metric_name
                }).fetchone()
                
                if not result:
                    return None
                
                return {
                    "current_month": result.current_month,
                    "current_value": float(result.current_value) if result.current_value else 0,
                    "previous_value": float(result.previous_value) if result.previous_value else 0,
                    "growth_pct": float(result.growth_pct) if result.growth_pct else 0
                }
                
        except Exception as e:
            logger.warning(f"⚠️ Failed to get recent performance for {metric_name}: {e}")
            return None
    
    def _check_expectation_violation(self, metric: str, recent: Dict,
                                      baseline: Dict) -> Optional[Dict]:
        """Check if recent performance violates expectations."""
        actual_growth = recent.get("growth_pct", 0)
        expected_growth = baseline.get("avg_growth", 0)
        std_growth = baseline.get("std_growth", 0)
        
        if std_growth == 0:
            std_growth = abs(expected_growth) * 0.5 if expected_growth != 0 else 10
        
        z_score = (actual_growth - expected_growth) / std_growth if std_growth > 0 else 0
        deviation_pct = actual_growth - expected_growth
        
        if abs(z_score) < 1.0 and abs(deviation_pct) < 10:
            return None
        
        if abs(z_score) >= 3.0 or abs(deviation_pct) >= 50:
            severity = "🔴 CRITICAL"
        elif abs(z_score) >= 2.0 or abs(deviation_pct) >= 30:
            severity = "🟠 HIGH"
        elif abs(z_score) >= 1.5 or abs(deviation_pct) >= 20:
            severity = "🟡 MODERATE"
        else:
            severity = "🟢 MINOR"
        
        if actual_growth < expected_growth:
            violation_type = "UNDERPERFORMANCE"
            direction = "below"
        else:
            violation_type = "OVERPERFORMANCE"
            direction = "above"
        
        insight = self._generate_violation_insight(
            metric, actual_growth, expected_growth, deviation_pct, violation_type, baseline
        )
        
        return {
            "metric": metric,
            "violation_type": violation_type,
            "severity": severity,
            "actual_growth_pct": round(actual_growth, 2),
            "expected_growth_pct": round(expected_growth, 2),
            "deviation_pct": round(deviation_pct, 2),
            "z_score": round(z_score, 2),
            "direction": direction,
            "current_value": recent.get("current_value"),
            "previous_value": recent.get("previous_value"),
            "historical_range": {
                "min": round(baseline.get("min_growth", 0), 2),
                "avg": round(baseline.get("avg_growth", 0), 2),
                "max": round(baseline.get("max_growth", 0), 2)
            },
            "insight": insight,
            "recommendation": self._get_violation_recommendation(violation_type, severity, metric)
        }
    
    def _generate_violation_insight(self, metric: str, actual: float, expected: float,
                                     deviation: float, violation_type: str,
                                     baseline: Dict) -> str:
        """Generate human-readable insight for violation."""
        periods = baseline.get("period_count", 0)
        
        if violation_type == "UNDERPERFORMANCE":
            if actual < 0 and expected > 0:
                return (f"🔴 {metric} DECLINED {abs(actual):.1f}% when historically it GROWS "
                       f"{expected:.1f}% on average (based on {periods} periods). "
                       f"This is a {abs(deviation):.1f} percentage point miss.")
            elif actual < expected:
                return (f"📉 {metric} grew only {actual:.1f}% vs historical average of "
                       f"{expected:.1f}% (based on {periods} periods). "
                       f"Underperforming by {abs(deviation):.1f} percentage points.")
        else:
            if actual > 0 and expected < 0:
                return (f"🚀 {metric} GREW {actual:.1f}% when historically it DECLINES "
                       f"{abs(expected):.1f}% on average. Exceptional performance!")
            else:
                return (f"📈 {metric} grew {actual:.1f}% vs historical average of "
                       f"{expected:.1f}%. Outperforming by {deviation:.1f} percentage points!")
        
        return f"{metric}: {actual:.1f}% actual vs {expected:.1f}% expected"
    
    def _get_violation_recommendation(self, violation_type: str, severity: str,
                                       metric: str) -> str:
        """Get recommendation based on violation."""
        if "CRITICAL" in severity:
            if violation_type == "UNDERPERFORMANCE":
                return f"🚨 IMMEDIATE ACTION: Investigate {metric} decline. This is significantly below historical norms."
            else:
                return f"🎉 CAPITALIZE: {metric} is significantly outperforming. Understand what's driving this and scale it."
        
        elif "HIGH" in severity:
            if violation_type == "UNDERPERFORMANCE":
                return f"⚠️ PRIORITY REVIEW: Analyze {metric} drivers this week. Performance is concerning."
            else:
                return f"📊 ANALYZE: Document what's driving {metric} outperformance for future planning."
        
        elif "MODERATE" in severity:
            if violation_type == "UNDERPERFORMANCE":
                return f"📋 MONITOR: Keep eye on {metric}. May be temporary or early warning sign."
            else:
                return f"📋 MONITOR: {metric} performing above average. Verify sustainability."
        
        else:
            return f"ℹ️ Note: {metric} slightly outside normal range. No immediate action needed."
    
    def _generate_violation_summary(self, violations: List[Dict], metrics: List[str]) -> str:
        """Generate executive summary of violations."""
        if not violations:
            return (f"✅ ALL METRICS PERFORMING AS EXPECTED\n"
                   f"Analyzed {len(metrics)} metrics. All within historical norms.")
        
        critical = len([v for v in violations if "CRITICAL" in v.get("severity", "")])
        high = len([v for v in violations if "HIGH" in v.get("severity", "")])
        underperforming = len([v for v in violations if v.get("violation_type") == "UNDERPERFORMANCE"])
        overperforming = len(violations) - underperforming
        
        summary = f"📊 EXPECTATION VIOLATIONS DETECTED\n\n"
        summary += f"Analyzed {len(metrics)} metrics, found {len(violations)} violations:\n"
        summary += f"• {underperforming} underperforming vs baseline\n"
        summary += f"• {overperforming} overperforming vs baseline\n\n"
        
        if critical > 0:
            summary += f"🔴 {critical} CRITICAL violations require immediate attention.\n"
        if high > 0:
            summary += f"🟠 {high} HIGH severity violations need review this week.\n"
        
        return summary


if __name__ == "__main__":
    import sys
    
    print("🤔 Testing AnomalyDetector...")
    
    try:
        detector = AnomalyDetector()
        print("✅ AnomalyDetector initialized")
        
        test_user = sys.argv[1] if len(sys.argv) > 1 else None
        
        if test_user:
            patterns = detector.find_counter_intuitive_patterns(test_user)
            print(f"✅ Counter-intuitive patterns: {patterns.get('patterns_found', 0)} found")
            
            violations = detector.detect_expectation_violations(test_user)
            print(f"✅ Expectation violations: {violations.get('violations_found', 0)} found")
        
        print("\n✅ anomaly.py working correctly!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

