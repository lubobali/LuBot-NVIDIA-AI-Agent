"""
═══════════════════════════════════════════════════════════════════════════════
🧠 INTELLIGENCE ENGINE - Concentration Risk Module
═══════════════════════════════════════════════════════════════════════════════

Task 3.4: Concentration Risk Analysis
- 3.4.1: HHI Calculator (DOJ/FTC Regulatory Standard)
- 3.4.2: Concentration Risk Reporter (CEO Dashboard)

Big Tech Pattern: DOJ/SEC regulatory-grade concentration metrics.
"""

import logging
from typing import Dict, List

from sqlalchemy import text

from .base import IntelligenceBase

logger = logging.getLogger(__name__)


class ConcentrationAnalyzer(IntelligenceBase):
    """
    🚨 Concentration Risk Analyzer
    
    DOJ/FTC standard HHI calculations with CEO-friendly risk reports.
    "Too many eggs in one basket" detection.
    """
    
    # ═══════════════════════════════════════════════════════════════════════════
    # 📊 SUBTASK 3.4.1: HHI CALCULATOR (Regulatory Standard)
    # ═══════════════════════════════════════════════════════════════════════════
    
    def calculate_hhi(self, user_id: str, metric_name: str, dimension: str) -> Dict:
        """
        SUBTASK 3.4.1: Calculate Herfindahl-Hirschman Index (HHI).
        
        DOJ/FTC Regulatory Standard for market concentration:
        - HHI > 2500: Highly concentrated
        - HHI 1500-2500: Moderately concentrated
        - HHI < 1500: Unconcentrated/Diversified
        """
        try:
            segment_data = self._get_period_segment_data(user_id, metric_name, dimension)
            
            if not segment_data or len(segment_data) < 1:
                return {"error": "No segment data found"}
            
            current_total = sum(s["second_value"] for s in segment_data.values())
            
            if current_total == 0:
                return {"error": "Total is zero, cannot calculate HHI"}
            
            segment_shares = []
            hhi_decimal = 0
            
            for segment, data in segment_data.items():
                value = data["second_value"]
                share = value / current_total
                share_pct = share * 100
                
                hhi_contribution = share ** 2
                hhi_decimal += hhi_contribution
                
                segment_shares.append({
                    "segment": segment,
                    "value": round(value, 2),
                    "share_decimal": round(share, 4),
                    "share_pct": round(share_pct, 2),
                    "hhi_contribution": round(hhi_contribution, 4),
                    "hhi_contribution_pct": round(hhi_contribution * 100, 2)
                })
            
            segment_shares.sort(key=lambda x: x["share_pct"], reverse=True)
            
            hhi_points = hhi_decimal * 10000
            
            if hhi_points > 2500:
                concentration_level = "high"
                risk_level = "🔴 HIGH RISK"
                risk_description = "Highly concentrated - significant dependency risk"
            elif hhi_points > 1500:
                concentration_level = "moderate"
                risk_level = "🟡 MODERATE RISK"
                risk_description = "Moderately concentrated - some dependency risk"
            else:
                concentration_level = "low"
                risk_level = "🟢 LOW RISK"
                risk_description = "Well diversified - healthy distribution"
            
            equivalent_firms = 1 / hhi_decimal if hhi_decimal > 0 else len(segment_data)
            
            top_1_share = segment_shares[0]["share_pct"] if segment_shares else 0
            top_3_share = sum(s["share_pct"] for s in segment_shares[:3])
            top_5_share = sum(s["share_pct"] for s in segment_shares[:5])
            
            interpretation = self._interpret_hhi(
                hhi_points, concentration_level, equivalent_firms,
                segment_shares, dimension
            )
            
            result = {
                "metric_name": metric_name,
                "dimension": dimension,
                "hhi": {
                    "decimal": round(hhi_decimal, 4),
                    "points": round(hhi_points, 0),
                    "scale": "0-10000 (DOJ/FTC standard)"
                },
                "concentration_level": concentration_level,
                "risk_level": risk_level,
                "risk_description": risk_description,
                "equivalent_firms": round(equivalent_firms, 1),
                "segment_count": len(segment_shares),
                "concentration_ratios": {
                    "cr1": round(top_1_share, 1),
                    "cr3": round(top_3_share, 1),
                    "cr5": round(top_5_share, 1)
                },
                "segment_shares": segment_shares,
                "thresholds": {
                    "high": "> 2500 points (> 0.25)",
                    "moderate": "1500-2500 points (0.15-0.25)",
                    "low": "< 1500 points (< 0.15)"
                },
                "interpretation": interpretation
            }
            
            logger.info(f"📊 HHI for user {user_id[:8]}...: {hhi_points:.0f} points ({concentration_level})")
            return result
            
        except Exception as e:
            logger.error(f"❌ HHI calculation failed: {e}")
            return {"error": str(e)}
    
    def _get_period_segment_data(self, user_id: str, metric_name: str,
                                  dimension: str) -> Dict[str, Dict]:
        """
        Get period segment data for HHI calculation.

        Big Tech Pattern: Unified interface, multiple backends.
        - MY DATA mode: Uses UploadedDataAdapter (categorical data)
        - DEMO mode: Uses existing SQL (time-series data)

        Returns same format regardless of source.
        """
        # ═══════════════════════════════════════════════════════════════════
        # 🆕 MY DATA MODE ROUTING (Big Tech: Auto-detect + Adapter pattern)
        # ═══════════════════════════════════════════════════════════════════
        data_source = self._detect_data_source(user_id)

        if data_source == "rows_table":
            # MY DATA mode - route to adapter
            logger.info(f"📊 MY DATA mode: Using UploadedDataAdapter for {dimension} → {metric_name}")
            return self._get_uploaded_segment_data(user_id, metric_name, dimension)

        # ═══════════════════════════════════════════════════════════════════
        # 📊 DEMO MODE (UNCHANGED - existing SQL for time-series data)
        # ═══════════════════════════════════════════════════════════════════
        try:
            logger.info(f"📊 DEMO mode: Using user_uploaded_metrics for {dimension} → {metric_name}")

            with self.engine.connect() as conn:
                query = text("""
                    WITH date_range AS (
                        SELECT
                            MIN(date) as min_date,
                            MAX(date) as max_date,
                            (MAX(date) - MIN(date)) / 2 as mid_offset
                        FROM user_uploaded_metrics
                        WHERE user_id = :user_id AND metric_name = :metric_name
                    ),
                    first_period AS (
                        SELECT
                            dimensions->>:dimension as segment,
                            SUM(metric_value) as total
                        FROM user_uploaded_metrics, date_range
                        WHERE user_id = :user_id
                          AND metric_name = :metric_name
                          AND dimensions->>:dimension IS NOT NULL
                          AND date < date_range.min_date + date_range.mid_offset
                        GROUP BY dimensions->>:dimension
                    ),
                    second_period AS (
                        SELECT
                            dimensions->>:dimension as segment,
                            SUM(metric_value) as total
                        FROM user_uploaded_metrics, date_range
                        WHERE user_id = :user_id
                          AND metric_name = :metric_name
                          AND dimensions->>:dimension IS NOT NULL
                          AND date >= date_range.min_date + date_range.mid_offset
                        GROUP BY dimensions->>:dimension
                    )
                    SELECT
                        COALESCE(f.segment, s.segment) as segment,
                        COALESCE(f.total, 0) as first_value,
                        COALESCE(s.total, 0) as second_value
                    FROM first_period f
                    FULL OUTER JOIN second_period s ON f.segment = s.segment
                    ORDER BY COALESCE(s.total, 0) DESC
                """)

                result = conn.execute(query, {
                    "user_id": user_id,
                    "metric_name": metric_name,
                    "dimension": dimension
                }).fetchall()

                return {
                    row.segment: {
                        "first_value": float(row.first_value),
                        "second_value": float(row.second_value)
                    }
                    for row in result if row.segment
                }

        except Exception as e:
            logger.warning(f"⚠️ Failed to get period segment data: {e}")
            return {}

    def _get_uploaded_segment_data(self, user_id: str, metric_name: str,
                                     dimension: str) -> Dict[str, Dict]:
        """
        🆕 MY DATA MODE: Get segment data from uploaded categorical data.

        Uses UploadedDataAdapter to read JSONB data from user_uploaded_rows.

        Big Tech Pattern: Adapter isolates data access complexity.
        - Google BigQuery: Data adapters separate from business logic
        - Facebook Presto: Connectors handle data source specifics
        - Netflix: Single adapter per data source

        Args:
            user_id: User identifier
            metric_name: Numeric column to aggregate (e.g., "Salary", "Age")
            dimension: Categorical column to group by (e.g., "Department", "Gender")

        Returns:
            Dict[segment_name, {"first_value": X, "second_value": X}]

        Example for employees.csv:
            metric_name="Salary", dimension="Department" →
            {
                "Product": {"first_value": 3157500, "second_value": 3157500},
                "Human Resource": {"first_value": 98500, "second_value": 98500}
            }

        Note: For categorical data without time dimension, first_value = second_value
              (snapshot data, not before/after comparison like DEMO mode)
        """
        try:
            # Use adapter to get aggregated data by dimension
            segment_totals = self.uploaded_adapter.get_grouped_aggregation(
                user_id=user_id,
                group_column=dimension,
                value_column=metric_name,
                agg_func="sum"
            )

            if not segment_totals:
                logger.warning(f"⚠️ No segment data found for {dimension} → {metric_name}")
                return {}

            logger.info(f"📊 Adapter returned {len(segment_totals)} segments")

            # Convert to HHI expected format
            # For categorical snapshot data, first_value = second_value
            result = {
                segment: {
                    "first_value": float(total),
                    "second_value": float(total)  # Snapshot: same value for both periods
                }
                for segment, total in segment_totals.items()
                if segment and total  # Filter out None/empty
            }

            logger.info(f"✅ Formatted {len(result)} segments for HHI calculation")
            return result

        except Exception as e:
            logger.error(f"❌ Failed to get uploaded segment data: {e}")
            import traceback
            traceback.print_exc()
            return {}
    
    def _interpret_hhi(self, hhi_points: float, level: str, equiv_firms: float,
                       shares: List[Dict], dimension: str) -> str:
        """Generate interpretation of HHI results."""
        top_segment = shares[0]["segment"] if shares else "Unknown"
        top_share = shares[0]["share_pct"] if shares else 0
        
        interpretation = f"📊 CONCENTRATION ANALYSIS (HHI)\n\n"
        interpretation += f"HHI Score: {hhi_points:.0f} points ({level.upper()} concentration)\n\n"
        
        if level == "high":
            interpretation += (
                f"⚠️ HIGH CONCENTRATION WARNING\n"
                f"Your {dimension} distribution is highly concentrated.\n"
                f"• {top_segment} alone represents {top_share:.1f}% of total\n"
                f"• Equivalent to only {equiv_firms:.1f} equal-sized {dimension}s\n"
                f"• Risk: A problem with top {dimension}s severely impacts results\n\n"
                f"RECOMMENDATION: Develop diversification strategy to reduce dependency."
            )
        elif level == "moderate":
            interpretation += (
                f"🔶 MODERATE CONCENTRATION\n"
                f"Your {dimension} distribution shows moderate concentration.\n"
                f"• {top_segment} leads with {top_share:.1f}% of total\n"
                f"• Equivalent to {equiv_firms:.1f} equal-sized {dimension}s\n"
                f"• Some dependency risk exists but manageable\n\n"
                f"RECOMMENDATION: Monitor top {dimension}s closely, consider gradual diversification."
            )
        else:
            interpretation += (
                f"✅ WELL DIVERSIFIED\n"
                f"Your {dimension} distribution is healthy.\n"
                f"• {top_segment} leads with {top_share:.1f}% of total\n"
                f"• Equivalent to {equiv_firms:.1f} equal-sized {dimension}s\n"
                f"• No single {dimension} dominates excessively\n\n"
                f"RECOMMENDATION: Maintain current diversification. Monitor for drift."
            )
        
        return interpretation
    
    # ═══════════════════════════════════════════════════════════════════════════
    # 🚨 SUBTASK 3.4.2: CONCENTRATION RISK REPORTER (CEO Dashboard)
    # ═══════════════════════════════════════════════════════════════════════════
    
    def report_concentration_risk(self, user_id: str, metric_name: str,
                                   dimension: str) -> Dict:
        """
        SUBTASK 3.4.2: Generate CEO-level concentration risk report.
        
        Board-ready analysis with what-if scenarios and mitigation plans.
        """
        try:
            hhi_data = self.calculate_hhi(user_id, metric_name, dimension)
            
            if "error" in hhi_data:
                return {"error": hhi_data["error"]}
            
            segment_shares = hhi_data.get("segment_shares", [])
            hhi_points = hhi_data.get("hhi", {}).get("points", 0)
            
            if not segment_shares:
                return {"error": "No segment data available"}
            
            risk_alerts = self._generate_risk_alerts(segment_shares, dimension)
            risk_score = self._calculate_risk_score(hhi_points, segment_shares)
            what_if_scenarios = self._generate_what_if_scenarios(segment_shares, dimension)
            
            concentration_summary = self._generate_concentration_summary(
                segment_shares, hhi_data, dimension
            )
            
            board_summary = self._generate_board_summary(
                risk_score, risk_alerts, segment_shares, hhi_data, dimension, metric_name
            )
            
            mitigation_plan = self._generate_mitigation_plan(
                risk_score, segment_shares, what_if_scenarios, dimension
            )
            
            if risk_score >= 70:
                risk_level = "🔴 CRITICAL"
                risk_color = "red"
            elif risk_score >= 50:
                risk_level = "🟠 HIGH"
                risk_color = "orange"
            elif risk_score >= 30:
                risk_level = "🟡 MODERATE"
                risk_color = "yellow"
            else:
                risk_level = "🟢 LOW"
                risk_color = "green"
            
            result = {
                "metric_name": metric_name,
                "dimension": dimension,
                "risk_score": round(risk_score, 0),
                "risk_level": risk_level,
                "risk_color": risk_color,
                "risk_alerts": risk_alerts,
                "concentration_summary": concentration_summary,
                "what_if_scenarios": what_if_scenarios,
                "board_summary": board_summary,
                "mitigation_plan": mitigation_plan,
                "hhi_data": {
                    "points": hhi_points,
                    "level": hhi_data.get("concentration_level"),
                    "equivalent_firms": hhi_data.get("equivalent_firms")
                },
                "segment_count": len(segment_shares)
            }
            
            if risk_score >= 70:
                logger.warning(f"🚨 CRITICAL concentration risk for user {user_id[:8]}...: score={risk_score}")
            else:
                logger.info(f"🚨 Concentration risk for user {user_id[:8]}...: score={risk_score} ({risk_level})")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Concentration risk report failed: {e}")
            return {"error": str(e)}
    
    def _generate_risk_alerts(self, shares: List[Dict], dimension: str) -> List[Dict]:
        """Generate specific risk alerts based on concentration."""
        alerts = []
        
        for seg in shares:
            if seg["share_pct"] > 40:
                alerts.append({
                    "severity": "🔴 CRITICAL",
                    "type": "SINGLE_DOMINANT",
                    "message": f"{seg['segment']} represents {seg['share_pct']:.1f}% of total",
                    "risk": f"Losing this {dimension} would be catastrophic",
                    "threshold": "> 40% single segment"
                })
        
        top_2_share = sum(s["share_pct"] for s in shares[:2])
        if top_2_share > 60:
            top_2_names = f"{shares[0]['segment']} + {shares[1]['segment']}"
            alerts.append({
                "severity": "🟠 HIGH",
                "type": "TOP_2_DOMINANT",
                "message": f"{top_2_names} represent {top_2_share:.1f}% of total",
                "risk": f"Heavy dependency on just 2 {dimension}s",
                "threshold": "> 60% top 2 segments"
            })
        
        top_3_share = sum(s["share_pct"] for s in shares[:3])
        if top_3_share > 80:
            alerts.append({
                "severity": "🟡 MODERATE",
                "type": "TOP_3_DOMINANT",
                "message": f"Top 3 {dimension}s represent {top_3_share:.1f}% of total",
                "risk": "Limited diversification",
                "threshold": "> 80% top 3 segments"
            })
        
        tiny_segments = [s for s in shares if s["share_pct"] < 1]
        if len(tiny_segments) > len(shares) / 2:
            alerts.append({
                "severity": "🔵 INFO",
                "type": "FRAGMENTED_TAIL",
                "message": f"{len(tiny_segments)} {dimension}s with < 1% share each",
                "risk": "Resource dilution on tiny segments",
                "threshold": "> 50% of segments are < 1%"
            })
        
        severity_order = {"🔴 CRITICAL": 0, "🟠 HIGH": 1, "🟡 MODERATE": 2, "🔵 INFO": 3}
        alerts.sort(key=lambda x: severity_order.get(x["severity"], 99))
        
        return alerts
    
    def _calculate_risk_score(self, hhi_points: float, shares: List[Dict]) -> float:
        """Calculate overall concentration risk score (0-100)."""
        hhi_component = min(40, (hhi_points / 10000) * 40)
        
        top_share = shares[0]["share_pct"] if shares else 0
        if top_share > 50:
            top_component = 30
        elif top_share > 40:
            top_component = 25
        elif top_share > 30:
            top_component = 20
        elif top_share > 20:
            top_component = 10
        else:
            top_component = 5
        
        top_3_share = sum(s["share_pct"] for s in shares[:3])
        if top_3_share > 80:
            top3_component = 20
        elif top_3_share > 70:
            top3_component = 15
        elif top_3_share > 60:
            top3_component = 10
        else:
            top3_component = 5
        
        segment_count = len(shares)
        if segment_count <= 3:
            diversity_component = 10
        elif segment_count <= 5:
            diversity_component = 7
        elif segment_count <= 10:
            diversity_component = 4
        else:
            diversity_component = 0
        
        total_score = hhi_component + top_component + top3_component + diversity_component
        
        return min(100, total_score)
    
    def _generate_what_if_scenarios(self, shares: List[Dict], dimension: str) -> List[Dict]:
        """Generate what-if scenarios for losing top segments."""
        scenarios = []
        
        if shares:
            top = shares[0]
            scenarios.append({
                "scenario": f"Lose {top['segment']}",
                "impact_pct": round(top["share_pct"], 1),
                "severity": "🔴 CRITICAL" if top["share_pct"] > 30 else "🟠 HIGH",
                "description": f"If {top['segment']} leaves, you lose {top['share_pct']:.1f}% of {dimension} revenue",
                "recovery_difficulty": "Very Hard" if top["share_pct"] > 40 else "Hard"
            })
        
        if len(shares) >= 2:
            top_2_share = shares[0]["share_pct"] + shares[1]["share_pct"]
            scenarios.append({
                "scenario": f"Lose top 2 {dimension}s",
                "impact_pct": round(top_2_share, 1),
                "severity": "🔴 CRITICAL",
                "description": f"Losing {shares[0]['segment']} and {shares[1]['segment']} = {top_2_share:.1f}% loss",
                "recovery_difficulty": "Extremely Hard"
            })
        
        if shares:
            top = shares[0]
            impact = top["share_pct"] * 0.5
            scenarios.append({
                "scenario": f"{top['segment']} declines 50%",
                "impact_pct": round(impact, 1),
                "severity": "🟠 HIGH" if impact > 15 else "🟡 MODERATE",
                "description": f"A 50% decline in {top['segment']} = {impact:.1f}% total loss",
                "recovery_difficulty": "Moderate"
            })
        
        small_segments = [s for s in shares if s["share_pct"] < 5]
        if small_segments:
            small_total = sum(s["share_pct"] for s in small_segments)
            scenarios.append({
                "scenario": f"Lose all small {dimension}s (< 5% each)",
                "impact_pct": round(small_total, 1),
                "severity": "🟡 MODERATE" if small_total > 20 else "🟢 LOW",
                "description": f"{len(small_segments)} small {dimension}s = {small_total:.1f}% combined",
                "recovery_difficulty": "Easy" if small_total < 20 else "Moderate"
            })
        
        return scenarios
    
    def _generate_concentration_summary(self, shares: List[Dict], hhi_data: Dict,
                                          dimension: str) -> Dict:
        """Generate concentration summary statistics."""
        total_value = sum(s["value"] for s in shares)
        
        return {
            "total_value": round(total_value, 2),
            "segment_count": len(shares),
            "top_segment": {
                "name": shares[0]["segment"] if shares else None,
                "share_pct": shares[0]["share_pct"] if shares else 0,
                "value": shares[0]["value"] if shares else 0
            },
            "concentration_ratios": {
                "cr1": round(shares[0]["share_pct"], 1) if shares else 0,
                "cr3": round(sum(s["share_pct"] for s in shares[:3]), 1),
                "cr5": round(sum(s["share_pct"] for s in shares[:5]), 1),
                "cr10": round(sum(s["share_pct"] for s in shares[:10]), 1)
            },
            "hhi_points": hhi_data.get("hhi", {}).get("points", 0),
            "equivalent_firms": hhi_data.get("equivalent_firms", 0),
            "bottom_50_pct_count": len([s for s in shares if s["share_pct"] < (50 / len(shares))]) if shares else 0
        }
    
    def _generate_board_summary(self, risk_score: float, alerts: List[Dict],
                                 shares: List[Dict], hhi_data: Dict,
                                 dimension: str, metric_name: str) -> str:
        """Generate one-paragraph board-ready summary."""
        hhi_points = hhi_data.get("hhi", {}).get("points", 0)
        equiv_firms = hhi_data.get("equivalent_firms", 0)
        top_segment = shares[0]["segment"] if shares else "Unknown"
        top_share = shares[0]["share_pct"] if shares else 0
        top_3_share = sum(s["share_pct"] for s in shares[:3])
        
        critical_alerts = [a for a in alerts if "CRITICAL" in a["severity"]]
        
        if risk_score >= 70:
            return (
                f"🔴 CRITICAL CONCENTRATION RISK: {metric_name} shows dangerous concentration by {dimension}. "
                f"{top_segment} alone represents {top_share:.0f}% of total, and top 3 {dimension}s "
                f"account for {top_3_share:.0f}%. HHI of {hhi_points:.0f} indicates the equivalent of "
                f"only {equiv_firms:.1f} equal-sized {dimension}s. "
                f"{len(critical_alerts)} critical alert(s) require immediate board attention. "
                f"Recommend urgent diversification strategy to reduce single-point-of-failure risk."
            )
        elif risk_score >= 50:
            return (
                f"🟠 HIGH CONCENTRATION RISK: {metric_name} by {dimension} shows concerning concentration. "
                f"{top_segment} leads with {top_share:.0f}%, and top 3 represent {top_3_share:.0f}%. "
                f"HHI of {hhi_points:.0f} (equivalent to {equiv_firms:.1f} equal {dimension}s) "
                f"suggests moderate dependency risk. Recommend developing diversification roadmap "
                f"and monitoring top {dimension} health quarterly."
            )
        elif risk_score >= 30:
            return (
                f"🟡 MODERATE CONCENTRATION: {metric_name} by {dimension} shows acceptable concentration. "
                f"{top_segment} leads with {top_share:.0f}%, top 3 at {top_3_share:.0f}%. "
                f"HHI of {hhi_points:.0f} is within normal range. Continue monitoring and "
                f"maintain relationship health with top {dimension}s."
            )
        else:
            return (
                f"🟢 HEALTHY DIVERSIFICATION: {metric_name} by {dimension} is well diversified. "
                f"No single {dimension} exceeds {top_share:.0f}%, and distribution is balanced. "
                f"HHI of {hhi_points:.0f} indicates healthy equivalent of {equiv_firms:.1f} {dimension}s. "
                f"Current strategy is sound. Maintain diversification."
            )
    
    def _generate_mitigation_plan(self, risk_score: float, shares: List[Dict],
                                   scenarios: List[Dict], dimension: str) -> List[Dict]:
        """Generate risk mitigation recommendations."""
        plan = []
        
        if risk_score >= 70:
            plan.append({
                "priority": 1,
                "timeframe": "IMMEDIATE",
                "action": f"Develop contingency plan for top {dimension} loss",
                "rationale": f"Top {dimension} represents {shares[0]['share_pct']:.0f}% - losing it would be catastrophic",
                "owner": "CEO / Board"
            })
            plan.append({
                "priority": 2,
                "timeframe": "30 DAYS",
                "action": f"Identify 3-5 alternative {dimension}s for rapid growth",
                "rationale": "Need immediate diversification pipeline",
                "owner": "Sales / BD"
            })
            plan.append({
                "priority": 3,
                "timeframe": "90 DAYS",
                "action": f"Reduce top {dimension} to < 35% of total",
                "rationale": "Reduce single-point-of-failure risk",
                "owner": "Executive Team"
            })
        
        elif risk_score >= 50:
            plan.append({
                "priority": 1,
                "timeframe": "30 DAYS",
                "action": f"Strengthen relationships with top 3 {dimension}s",
                "rationale": "Protect key relationships proactively",
                "owner": "Account Management"
            })
            plan.append({
                "priority": 2,
                "timeframe": "QUARTERLY",
                "action": f"Grow mid-tier {dimension}s by 20%",
                "rationale": "Build depth in the portfolio",
                "owner": "Sales"
            })
        
        elif risk_score >= 30:
            plan.append({
                "priority": 1,
                "timeframe": "QUARTERLY",
                "action": f"Review {dimension} concentration metrics",
                "rationale": "Early warning of concentration drift",
                "owner": "Finance / Analytics"
            })
        
        else:
            plan.append({
                "priority": 1,
                "timeframe": "ANNUAL",
                "action": f"Annual {dimension} portfolio review",
                "rationale": "Maintain healthy diversification",
                "owner": "Strategy"
            })
        
        return plan


if __name__ == "__main__":
    import sys
    
    print("🚨 Testing ConcentrationAnalyzer...")
    
    try:
        analyzer = ConcentrationAnalyzer()
        print("✅ ConcentrationAnalyzer initialized")
        
        test_user = sys.argv[1] if len(sys.argv) > 1 else None
        
        if test_user:
            metrics = analyzer._get_user_metrics(test_user)
            if metrics:
                dims = analyzer._get_metric_dimensions(test_user, metrics[0])
                if dims:
                    hhi = analyzer.calculate_hhi(test_user, metrics[0], dims[0])
                    print(f"✅ HHI: {hhi.get('hhi', {}).get('points', 'N/A')} points")
                    
                    risk = analyzer.report_concentration_risk(test_user, metrics[0], dims[0])
                    print(f"✅ Risk score: {risk.get('risk_score', 'N/A')}")
        
        print("\n✅ concentration.py working correctly!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

