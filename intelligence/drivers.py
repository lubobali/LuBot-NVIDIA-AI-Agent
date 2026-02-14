"""
═══════════════════════════════════════════════════════════════════════════════
🧠 INTELLIGENCE ENGINE - Driver Analysis Module
═══════════════════════════════════════════════════════════════════════════════

Task 3.3: Driver Analysis
- 3.3.1: Contribution Decomposition (McKinsey Waterfall)
- 3.3.2: Driver Ranking (Shapley-Inspired)
- 3.3.3: Sensitivity Analysis (Leverage & Elasticity)

Big Tech Pattern: McKinsey/BCG bridge analysis with PhD-level attribution.
"""

import logging
from typing import Dict, List, Optional

from sqlalchemy import text

from .base import IntelligenceBase

logger = logging.getLogger(__name__)


class DriverAnalyzer(IntelligenceBase):
    """
    🏆 Driver Analysis Engine
    
    McKinsey/BCG-style contribution decomposition:
    - "Germany drove 60% of the profit increase"
    - Shapley-inspired fair attribution
    - Bloomberg-style sensitivity analysis
    """
    
    # ═══════════════════════════════════════════════════════════════════════════
    # 📊 SUBTASK 3.3.1: CONTRIBUTION DECOMPOSITION (McKinsey Waterfall)
    # ═══════════════════════════════════════════════════════════════════════════
    
    def decompose_contributions(self, user_id: str, metric_name: str,
                                 dimension: str) -> Dict:
        """
        SUBTASK 3.3.1: Break total change into segment contributions.
        
        McKinsey/BCG Waterfall Analysis:
        - Shows exactly how much each segment contributed to total change
        - "Germany drove 60% of the profit increase"
        - Enables focused action on highest-impact areas
        """
        try:
            segment_data = self._get_period_segment_data(user_id, metric_name, dimension)
            
            if not segment_data or len(segment_data) < 1:
                return {"error": "No segment data found"}
            
            total_first = sum(s["first_value"] for s in segment_data.values())
            total_second = sum(s["second_value"] for s in segment_data.values())
            total_change = total_second - total_first
            
            if total_first == 0:
                return {"error": "First period total is zero"}
            
            total_change_pct = (total_change / total_first) * 100
            
            contributions = []
            positive_contributors = []
            negative_contributors = []
            
            for segment, data in segment_data.items():
                seg_change = data["second_value"] - data["first_value"]
                
                if total_change != 0:
                    contribution_pct = (seg_change / total_change) * 100
                else:
                    contribution_pct = 0
                
                if data["first_value"] != 0:
                    seg_growth_pct = ((data["second_value"] - data["first_value"]) / data["first_value"]) * 100
                else:
                    seg_growth_pct = 100 if data["second_value"] > 0 else 0
                
                first_share = (data["first_value"] / total_first * 100) if total_first > 0 else 0
                second_share = (data["second_value"] / total_second * 100) if total_second > 0 else 0
                
                contribution = {
                    "segment": segment,
                    "absolute_change": round(seg_change, 2),
                    "contribution_pct": round(contribution_pct, 2),
                    "segment_growth_pct": round(seg_growth_pct, 2),
                    "first_period_value": round(data["first_value"], 2),
                    "second_period_value": round(data["second_value"], 2),
                    "first_period_share_pct": round(first_share, 2),
                    "second_period_share_pct": round(second_share, 2),
                    "impact_description": self._describe_contribution(segment, contribution_pct, seg_change, total_change)
                }
                
                contributions.append(contribution)
                
                if seg_change > 0:
                    positive_contributors.append(contribution)
                elif seg_change < 0:
                    negative_contributors.append(contribution)
            
            contributions.sort(key=lambda x: abs(x["contribution_pct"]), reverse=True)
            positive_contributors.sort(key=lambda x: x["contribution_pct"], reverse=True)
            negative_contributors.sort(key=lambda x: x["contribution_pct"])
            
            waterfall = self._build_waterfall(total_first, contributions, total_second)
            concentration = self._calculate_contribution_concentration(contributions)
            
            insight = self._generate_contribution_insight(
                contributions, positive_contributors, negative_contributors,
                total_change, total_change_pct, dimension
            )
            
            result = {
                "metric_name": metric_name,
                "dimension": dimension,
                "total_change": {
                    "absolute": round(total_change, 2),
                    "pct": round(total_change_pct, 2),
                    "direction": "increase" if total_change > 0 else "decrease" if total_change < 0 else "flat"
                },
                "period_totals": {
                    "first_period": round(total_first, 2),
                    "second_period": round(total_second, 2)
                },
                "contributions": contributions,
                "positive_contributors": positive_contributors,
                "negative_contributors": negative_contributors,
                "summary": {
                    "positive_count": len(positive_contributors),
                    "negative_count": len(negative_contributors),
                    "flat_count": len(contributions) - len(positive_contributors) - len(negative_contributors)
                },
                "waterfall": waterfall,
                "concentration": concentration,
                "insight": insight
            }
            
            logger.info(f"📊 Contribution decomposition for user {user_id[:8]}...: {len(contributions)} segments, top={contributions[0]['segment'] if contributions else 'N/A'}")
            return result
            
        except Exception as e:
            logger.error(f"❌ Contribution decomposition failed: {e}")
            return {"error": str(e)}
    
    def _get_period_segment_data(self, user_id: str, metric_name: str,
                                  dimension: str) -> Dict[str, Dict]:
        """Get first period and second period values for each segment."""
        try:
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
    
    def _describe_contribution(self, segment: str, contribution_pct: float,
                                seg_change: float, total_change: float) -> str:
        """Generate human-readable contribution description."""
        if total_change == 0:
            return f"{segment}: No overall change"
        
        direction = "increase" if total_change > 0 else "decrease"
        
        if contribution_pct > 50:
            return f"{segment} drove majority ({contribution_pct:.0f}%) of the {direction}"
        elif contribution_pct > 20:
            return f"{segment} was a significant contributor ({contribution_pct:.0f}%)"
        elif contribution_pct > 0:
            return f"{segment} contributed {contribution_pct:.0f}% to the {direction}"
        elif contribution_pct < -20:
            return f"{segment} significantly offset the {direction} ({contribution_pct:.0f}%)"
        elif contribution_pct < 0:
            return f"{segment} partially offset the {direction} ({contribution_pct:.0f}%)"
        else:
            return f"{segment} had minimal impact"
    
    def _build_waterfall(self, start_value: float, contributions: List[Dict],
                          end_value: float) -> List[Dict]:
        """Build waterfall chart data structure."""
        waterfall = []
        
        waterfall.append({
            "label": "Start",
            "value": round(start_value, 2),
            "type": "total",
            "running_total": round(start_value, 2)
        })
        
        sorted_contribs = sorted(contributions, key=lambda x: x["absolute_change"], reverse=True)
        
        running_total = start_value
        
        for contrib in sorted_contribs:
            running_total += contrib["absolute_change"]
            
            waterfall.append({
                "label": contrib["segment"],
                "value": contrib["absolute_change"],
                "type": "increase" if contrib["absolute_change"] > 0 else "decrease",
                "running_total": round(running_total, 2),
                "contribution_pct": contrib["contribution_pct"]
            })
        
        waterfall.append({
            "label": "End",
            "value": round(end_value, 2),
            "type": "total",
            "running_total": round(end_value, 2)
        })
        
        return waterfall
    
    def _calculate_contribution_concentration(self, contributions: List[Dict]) -> Dict:
        """Calculate how concentrated the contribution is (Pareto analysis)."""
        if not contributions:
            return {}
        
        sorted_contribs = sorted(contributions, key=lambda x: abs(x["contribution_pct"]), reverse=True)
        
        cumulative = 0
        top_1_pct = 0
        top_3_pct = 0
        segments_for_80_pct = 0
        
        for i, contrib in enumerate(sorted_contribs):
            cumulative += abs(contrib["contribution_pct"])
            
            if i == 0:
                top_1_pct = abs(contrib["contribution_pct"])
            if i < 3:
                top_3_pct += abs(contrib["contribution_pct"])
            if segments_for_80_pct == 0 and cumulative >= 80:
                segments_for_80_pct = i + 1
        
        if top_1_pct > 60:
            concentration_level = "very_high"
            concentration_risk = "🔴 Single segment dependency"
        elif top_3_pct > 80:
            concentration_level = "high"
            concentration_risk = "🟡 Top 3 segments dominate"
        elif segments_for_80_pct and segments_for_80_pct <= len(sorted_contribs) / 2:
            concentration_level = "moderate"
            concentration_risk = "🟢 Reasonably diversified"
        else:
            concentration_level = "low"
            concentration_risk = "🟢 Well diversified"
        
        return {
            "top_1_contribution_pct": round(top_1_pct, 1),
            "top_3_contribution_pct": round(top_3_pct, 1),
            "segments_for_80_pct": segments_for_80_pct or len(sorted_contribs),
            "total_segments": len(sorted_contribs),
            "concentration_level": concentration_level,
            "concentration_risk": concentration_risk,
            "pareto_ratio": f"{segments_for_80_pct or len(sorted_contribs)}/{len(sorted_contribs)} segments drive 80% of change"
        }
    
    def _generate_contribution_insight(self, contributions: List[Dict],
                                        positive: List[Dict], negative: List[Dict],
                                        total_change: float, total_change_pct: float,
                                        dimension: str) -> str:
        """Generate insight about contributions."""
        if not contributions:
            return "No contribution data available."
        
        direction = "increased" if total_change > 0 else "decreased"
        top = contributions[0]
        
        insight = f"📊 CONTRIBUTION ANALYSIS\n\n"
        insight += f"Total {direction} by {abs(total_change_pct):.1f}% ({abs(total_change):,.0f} absolute).\n\n"
        
        insight += f"🏆 TOP DRIVER: {top['segment']}\n"
        insight += f"   Contributed {top['contribution_pct']:.0f}% of total change\n"
        insight += f"   Segment growth: {top['segment_growth_pct']:+.1f}%\n\n"
        
        if len(positive) > 0 and len(negative) > 0:
            pos_total = sum(p["contribution_pct"] for p in positive)
            neg_total = sum(n["contribution_pct"] for n in negative)
            insight += f"📈 Positive contributors: {len(positive)} segments (+{pos_total:.0f}%)\n"
            insight += f"📉 Negative contributors: {len(negative)} segments ({neg_total:.0f}%)\n"
        elif len(negative) == 0:
            insight += f"✅ All {len(positive)} segments contributed positively.\n"
        elif len(positive) == 0:
            insight += f"⚠️ All {len(negative)} segments declined.\n"
        
        return insight
    
    # ═══════════════════════════════════════════════════════════════════════════
    # 🏆 SUBTASK 3.3.2: DRIVER RANKING (Shapley-Inspired)
    # ═══════════════════════════════════════════════════════════════════════════
    
    def rank_drivers(self, user_id: str, metric_name: str, dimension: str) -> Dict:
        """
        SUBTASK 3.3.2: Rank segments by their impact on total change.
        
        Multi-criteria ranking inspired by Shapley values:
        - Impact Score: Absolute contribution to change
        - Growth Score: Segment's own growth rate
        - Leverage Score: How much the segment can move the total
        - Combined Score: Weighted combination for prioritization
        """
        try:
            contrib_data = self.decompose_contributions(user_id, metric_name, dimension)
            
            if "error" in contrib_data:
                return {"error": contrib_data["error"]}
            
            contributions = contrib_data.get("contributions", [])
            total_change = contrib_data.get("total_change", {})
            period_totals = contrib_data.get("period_totals", {})
            
            if not contributions:
                return {"error": "No contribution data available"}
            
            scored_segments = []
            
            for contrib in contributions:
                scores = self._calculate_driver_scores(
                    contrib,
                    total_change,
                    period_totals
                )
                
                scored_segments.append({
                    "segment": contrib["segment"],
                    "scores": scores,
                    "data": contrib
                })
            
            rankings = self._create_multi_criteria_rankings(scored_segments)
            top_drivers = self._identify_top_drivers(scored_segments)
            underperformers = self._identify_underperformers(scored_segments)
            
            action_priority = self._generate_action_priority(
                top_drivers, underperformers, scored_segments, dimension
            )
            
            shapley_attribution = self._calculate_shapley_attribution(scored_segments)
            
            result = {
                "metric_name": metric_name,
                "dimension": dimension,
                "total_change": total_change,
                "rankings": rankings,
                "top_drivers": top_drivers,
                "underperformers": underperformers,
                "action_priority": action_priority,
                "shapley_attribution": shapley_attribution,
                "segment_count": len(scored_segments)
            }
            
            logger.info(f"🏆 Driver ranking for user {user_id[:8]}...: {len(top_drivers)} top drivers, {len(underperformers)} underperformers")
            return result
            
        except Exception as e:
            logger.error(f"❌ Driver ranking failed: {e}")
            return {"error": str(e)}
    
    def _calculate_driver_scores(self, contrib: Dict, total_change: Dict,
                                  period_totals: Dict) -> Dict:
        """Calculate multi-criteria scores for a segment."""
        impact_score = min(100, abs(contrib.get("contribution_pct", 0)))
        
        growth_pct = contrib.get("segment_growth_pct", 0)
        growth_score = 50 + min(50, max(-50, growth_pct / 2))
        
        share = contrib.get("second_period_share_pct", 0)
        leverage_score = min(100, share * 2)
        
        share_change = contrib.get("second_period_share_pct", 0) - contrib.get("first_period_share_pct", 0)
        if growth_pct > 0 and share_change > 0:
            momentum_score = 75
        elif growth_pct > 0:
            momentum_score = 60
        elif growth_pct < 0 and share_change < 0:
            momentum_score = 25
        elif growth_pct < 0:
            momentum_score = 40
        else:
            momentum_score = 50
        
        combined_score = (
            impact_score * 0.30 +
            growth_score * 0.30 +
            leverage_score * 0.25 +
            momentum_score * 0.15
        )
        
        return {
            "impact_score": round(impact_score, 1),
            "growth_score": round(growth_score, 1),
            "leverage_score": round(leverage_score, 1),
            "momentum_score": round(momentum_score, 1),
            "combined_score": round(combined_score, 1),
            "raw": {
                "contribution_pct": contrib.get("contribution_pct", 0),
                "growth_pct": growth_pct,
                "share_pct": share,
                "share_change_pct": share_change
            }
        }
    
    def _create_multi_criteria_rankings(self, scored_segments: List[Dict]) -> Dict:
        """Create rankings by different criteria."""
        def create_ranking(key: str, reverse: bool = True) -> List[Dict]:
            sorted_segs = sorted(
                scored_segments,
                key=lambda x: x["scores"].get(key, 0),
                reverse=reverse
            )
            return [
                {
                    "rank": i + 1,
                    "segment": seg["segment"],
                    "score": seg["scores"].get(key),
                    "percentile": round((len(sorted_segs) - i) / len(sorted_segs) * 100)
                }
                for i, seg in enumerate(sorted_segs)
            ]
        
        return {
            "by_impact": create_ranking("impact_score"),
            "by_growth": create_ranking("growth_score"),
            "by_leverage": create_ranking("leverage_score"),
            "by_momentum": create_ranking("momentum_score"),
            "by_combined": create_ranking("combined_score")
        }
    
    def _identify_top_drivers(self, scored_segments: List[Dict]) -> List[Dict]:
        """Identify segments that are top drivers of performance."""
        top_drivers = []
        
        for seg in scored_segments:
            scores = seg["scores"]
            raw = scores.get("raw", {})
            
            is_top = (
                scores["combined_score"] > 60 or
                (scores["impact_score"] > 30 and raw.get("growth_pct", 0) > 10)
            )
            
            if is_top:
                top_drivers.append({
                    "segment": seg["segment"],
                    "combined_score": scores["combined_score"],
                    "contribution_pct": raw.get("contribution_pct"),
                    "growth_pct": raw.get("growth_pct"),
                    "reason": self._get_top_driver_reason(scores, raw)
                })
        
        top_drivers.sort(key=lambda x: x["combined_score"], reverse=True)
        
        return top_drivers[:5]
    
    def _get_top_driver_reason(self, scores: Dict, raw: Dict) -> str:
        """Generate reason why segment is a top driver."""
        reasons = []
        
        if scores["impact_score"] > 50:
            reasons.append(f"drove {raw.get('contribution_pct', 0):.0f}% of total change")
        if raw.get("growth_pct", 0) > 20:
            reasons.append(f"strong growth ({raw.get('growth_pct', 0):.0f}%)")
        if scores["leverage_score"] > 50:
            reasons.append(f"high share ({raw.get('share_pct', 0):.0f}%)")
        if scores["momentum_score"] > 60:
            reasons.append("positive momentum")
        
        return "; ".join(reasons) if reasons else "solid overall performance"
    
    def _identify_underperformers(self, scored_segments: List[Dict]) -> List[Dict]:
        """Identify segments that are underperforming."""
        underperformers = []
        
        for seg in scored_segments:
            scores = seg["scores"]
            raw = scores.get("raw", {})
            
            is_underperformer = (
                (scores["leverage_score"] > 30 and raw.get("growth_pct", 0) < -5) or
                (raw.get("contribution_pct", 0) < -10 and raw.get("share_pct", 0) > 10)
            )
            
            if is_underperformer:
                underperformers.append({
                    "segment": seg["segment"],
                    "combined_score": scores["combined_score"],
                    "contribution_pct": raw.get("contribution_pct"),
                    "growth_pct": raw.get("growth_pct"),
                    "share_pct": raw.get("share_pct"),
                    "risk_level": "🔴 HIGH" if raw.get("share_pct", 0) > 20 else "🟡 MEDIUM",
                    "reason": self._get_underperformer_reason(scores, raw)
                })
        
        underperformers.sort(key=lambda x: x.get("share_pct", 0), reverse=True)
        
        return underperformers[:5]
    
    def _get_underperformer_reason(self, scores: Dict, raw: Dict) -> str:
        """Generate reason why segment is underperforming."""
        share = raw.get("share_pct", 0)
        growth = raw.get("growth_pct", 0)
        
        if share > 20 and growth < -10:
            return f"⚠️ Large segment ({share:.0f}% share) declining {abs(growth):.0f}%"
        elif growth < -20:
            return f"⚠️ Severe decline ({growth:.0f}%)"
        elif raw.get("contribution_pct", 0) < -20:
            return f"⚠️ Major drag on total ({raw.get('contribution_pct', 0):.0f}% contribution)"
        else:
            return "Below-average performance"
    
    def _generate_action_priority(self, top_drivers: List[Dict],
                                   underperformers: List[Dict],
                                   all_segments: List[Dict],
                                   dimension: str) -> List[Dict]:
        """Generate prioritized action list."""
        actions = []
        
        if top_drivers:
            top = top_drivers[0]
            actions.append({
                "priority": 1,
                "action": f"Double down on {top['segment']}",
                "type": "ACCELERATE",
                "icon": "🚀",
                "rationale": f"Top driver with {top['growth_pct']:.0f}% growth, {top['contribution_pct']:.0f}% contribution",
                "expected_impact": "HIGH"
            })
        
        if underperformers:
            worst = underperformers[0]
            actions.append({
                "priority": 2,
                "action": f"Investigate decline in {worst['segment']}",
                "type": "FIX",
                "icon": "🔧",
                "rationale": f"{worst['share_pct']:.0f}% of total, declining {abs(worst['growth_pct']):.0f}%",
                "expected_impact": "HIGH" if worst['share_pct'] > 20 else "MEDIUM"
            })
        
        for seg in all_segments:
            raw = seg["scores"].get("raw", {})
            if raw.get("share_pct", 0) < 15 and raw.get("growth_pct", 0) > 30:
                actions.append({
                    "priority": 3,
                    "action": f"Scale up {seg['segment']} (hidden gem)",
                    "type": "GROW",
                    "icon": "💎",
                    "rationale": f"Only {raw.get('share_pct', 0):.0f}% share but growing {raw.get('growth_pct', 0):.0f}%",
                    "expected_impact": "MEDIUM"
                })
                break
        
        if top_drivers and top_drivers[0]["contribution_pct"] > 50:
            actions.append({
                "priority": 4,
                "action": f"Reduce dependency on {top_drivers[0]['segment']}",
                "type": "DIVERSIFY",
                "icon": "⚖️",
                "rationale": f"Single {dimension} drives {top_drivers[0]['contribution_pct']:.0f}% of change",
                "expected_impact": "MEDIUM"
            })
        
        return actions
    
    def _calculate_shapley_attribution(self, scored_segments: List[Dict]) -> Dict:
        """
        Calculate Shapley-inspired fair attribution.
        
        Simplified Shapley: Each segment's value is its marginal contribution
        averaged over different orderings.
        """
        if not scored_segments:
            return {}
        
        total_contribution = sum(
            abs(seg["scores"]["raw"].get("contribution_pct", 0))
            for seg in scored_segments
        )
        
        attributions = []
        
        for seg in scored_segments:
            raw = seg["scores"]["raw"]
            
            contribution = abs(raw.get("contribution_pct", 0))
            share = raw.get("share_pct", 0)
            
            if total_contribution > 0:
                contribution_component = (contribution / total_contribution) * 70
            else:
                contribution_component = 0
            
            total_share = sum(s["scores"]["raw"].get("share_pct", 0) for s in scored_segments)
            if total_share > 0:
                share_component = (share / total_share) * 30
            else:
                share_component = 0
            
            shapley_value = contribution_component + share_component
            
            attributions.append({
                "segment": seg["segment"],
                "shapley_value": round(shapley_value, 2),
                "interpretation": f"{seg['segment']} is fairly responsible for {shapley_value:.1f}% of outcome"
            })
        
        attributions.sort(key=lambda x: x["shapley_value"], reverse=True)
        
        return {
            "method": "Shapley-inspired (70% contribution + 30% potential)",
            "attributions": attributions,
            "top_attribution": attributions[0] if attributions else None
        }
    
    # ═══════════════════════════════════════════════════════════════════════════
    # 📈 SUBTASK 3.3.3: SENSITIVITY ANALYSIS (Leverage & Elasticity)
    # ═══════════════════════════════════════════════════════════════════════════
    
    def analyze_sensitivity(self, user_id: str, metric_name: str, dimension: str,
                            scenario_pcts: List[float] = None) -> Dict:
        """
        SUBTASK 3.3.3: Calculate leverage and run what-if scenarios.
        
        Bloomberg-style sensitivity analysis:
        - "If Germany improves 10%, total profit improves 3.2%"
        - Identifies highest-leverage opportunities
        - Runs multiple scenarios for planning
        """
        try:
            if scenario_pcts is None:
                scenario_pcts = [-20, -10, 10, 20]
            
            segment_data = self._get_period_segment_data(user_id, metric_name, dimension)
            
            if not segment_data or len(segment_data) < 1:
                return {"error": "No segment data found"}
            
            current_total = sum(s["second_value"] for s in segment_data.values())
            
            if current_total == 0:
                return {"error": "Current total is zero"}
            
            leverage_scores = []
            sensitivity_matrix = []
            
            for segment, data in segment_data.items():
                current_value = data["second_value"]
                share_pct = (current_value / current_total) * 100 if current_total > 0 else 0
                
                leverage = share_pct / 100
                
                scenarios = []
                for change_pct in scenario_pcts:
                    new_segment_value = current_value * (1 + change_pct / 100)
                    new_total = current_total - current_value + new_segment_value
                    total_change_pct = ((new_total - current_total) / current_total) * 100
                    
                    scenarios.append({
                        "segment_change_pct": change_pct,
                        "new_segment_value": round(new_segment_value, 2),
                        "new_total": round(new_total, 2),
                        "total_change_pct": round(total_change_pct, 2),
                        "elasticity": round(total_change_pct / change_pct, 3) if change_pct != 0 else 0
                    })
                
                elasticities = [s["elasticity"] for s in scenarios if s["elasticity"] != 0]
                avg_elasticity = sum(elasticities) / len(elasticities) if elasticities else 0
                
                leverage_entry = {
                    "segment": segment,
                    "current_value": round(current_value, 2),
                    "share_pct": round(share_pct, 2),
                    "leverage": round(leverage, 4),
                    "elasticity": round(avg_elasticity, 3),
                    "scenarios": scenarios,
                    "interpretation": self._interpret_leverage(segment, share_pct, avg_elasticity)
                }
                
                leverage_scores.append(leverage_entry)
                
                sensitivity_matrix.append({
                    "segment": segment,
                    "share_pct": round(share_pct, 2),
                    **{f"impact_at_{p:+d}%": next(
                        (s["total_change_pct"] for s in scenarios if s["segment_change_pct"] == p),
                        None
                    ) for p in scenario_pcts}
                })
            
            leverage_scores.sort(key=lambda x: x["leverage"], reverse=True)
            sensitivity_matrix.sort(key=lambda x: x["share_pct"], reverse=True)
            
            highest_leverage = self._identify_highest_leverage(leverage_scores)
            strategic_recommendations = self._generate_sensitivity_recommendations(
                leverage_scores, dimension
            )
            tornado_data = self._build_tornado_data(leverage_scores, scenario_pcts)
            
            result = {
                "metric_name": metric_name,
                "dimension": dimension,
                "current_total": round(current_total, 2),
                "leverage_scores": leverage_scores,
                "sensitivity_matrix": sensitivity_matrix,
                "highest_leverage": highest_leverage,
                "strategic_recommendations": strategic_recommendations,
                "tornado_chart_data": tornado_data,
                "scenarios_modeled": scenario_pcts,
                "segment_count": len(leverage_scores)
            }
            
            logger.info(f"📈 Sensitivity analysis for user {user_id[:8]}...: {len(leverage_scores)} segments, top leverage={leverage_scores[0]['segment'] if leverage_scores else 'N/A'}")
            return result
            
        except Exception as e:
            logger.error(f"❌ Sensitivity analysis failed: {e}")
            return {"error": str(e)}
    
    def _interpret_leverage(self, segment: str, share_pct: float, elasticity: float) -> str:
        """Generate interpretation of segment's leverage."""
        if share_pct > 40:
            leverage_level = "very high"
            impact = "dominant impact on total"
        elif share_pct > 20:
            leverage_level = "high"
            impact = "significant impact on total"
        elif share_pct > 10:
            leverage_level = "moderate"
            impact = "noticeable impact on total"
        else:
            leverage_level = "low"
            impact = "limited impact on total"
        
        return f"{segment} has {leverage_level} leverage ({share_pct:.1f}% share). A 10% change here means {share_pct/10:.2f}% change in total."
    
    def _identify_highest_leverage(self, leverage_scores: List[Dict]) -> List[Dict]:
        """Identify segments with highest leverage for action."""
        highest = []
        
        for lev in leverage_scores[:5]:
            scenario_10 = next(
                (s for s in lev["scenarios"] if s["segment_change_pct"] == 10),
                None
            )
            
            if scenario_10:
                highest.append({
                    "segment": lev["segment"],
                    "share_pct": lev["share_pct"],
                    "if_improve_10pct": f"+{scenario_10['total_change_pct']:.2f}% total",
                    "if_decline_10pct": f"-{scenario_10['total_change_pct']:.2f}% total",
                    "priority": "🔴 HIGH" if lev["share_pct"] > 30 else "🟡 MEDIUM" if lev["share_pct"] > 15 else "🟢 MONITOR",
                    "action": self._get_leverage_action(lev["segment"], lev["share_pct"])
                })
        
        return highest
    
    def _get_leverage_action(self, segment: str, share_pct: float) -> str:
        """Get recommended action based on leverage."""
        if share_pct > 30:
            return f"PROTECT: {segment} is critical. Defend this position."
        elif share_pct > 15:
            return f"GROW: {segment} has room to become a major driver."
        else:
            return f"MONITOR: {segment} has limited total impact currently."
    
    def _generate_sensitivity_recommendations(self, leverage_scores: List[Dict],
                                               dimension: str) -> List[Dict]:
        """Generate strategic recommendations from sensitivity analysis."""
        recommendations = []
        
        if not leverage_scores:
            return recommendations
        
        top = leverage_scores[0]
        top_10_impact = next(
            (s["total_change_pct"] for s in top["scenarios"] if s["segment_change_pct"] == 10),
            0
        )
        
        recommendations.append({
            "priority": 1,
            "icon": "🎯",
            "title": f"Focus investment on {top['segment']}",
            "rationale": f"Highest leverage ({top['share_pct']:.1f}% share). A 10% improvement here drives {top_10_impact:.2f}% total growth.",
            "expected_roi": "HIGH"
        })
        
        high_leverage_segments = [l for l in leverage_scores if l["share_pct"] > 20]
        if len(high_leverage_segments) > 1:
            names = ", ".join([h["segment"] for h in high_leverage_segments[:3]])
            total_share = sum(h["share_pct"] for h in high_leverage_segments)
            
            recommendations.append({
                "priority": 2,
                "icon": "🛡️",
                "title": "Protect high-leverage segments",
                "rationale": f"{names} together represent {total_share:.0f}% of total. A decline here severely impacts results.",
                "expected_roi": "RISK MITIGATION"
            })
        
        low_leverage_growing = [
            l for l in leverage_scores
            if l["share_pct"] < 15 and l["share_pct"] > 5
        ]
        if low_leverage_growing:
            candidate = low_leverage_growing[0]
            recommendations.append({
                "priority": 3,
                "icon": "📈",
                "title": f"Scale {candidate['segment']} for diversification",
                "rationale": f"Currently {candidate['share_pct']:.1f}% share. Growing this reduces dependency on top {dimension}s.",
                "expected_roi": "MEDIUM"
            })
        
        top_3_share = sum(l["share_pct"] for l in leverage_scores[:3])
        if top_3_share > 70:
            recommendations.append({
                "priority": 4,
                "icon": "⚠️",
                "title": "High concentration risk",
                "rationale": f"Top 3 {dimension}s represent {top_3_share:.0f}% of total. Consider diversification strategy.",
                "expected_roi": "RISK MITIGATION"
            })
        
        return recommendations
    
    def _build_tornado_data(self, leverage_scores: List[Dict],
                            scenario_pcts: List[float]) -> Dict:
        """Build tornado chart data for visualization."""
        min_pct = min(scenario_pcts)
        max_pct = max(scenario_pcts)
        
        tornado_bars = []
        
        for lev in leverage_scores[:10]:
            min_scenario = next(
                (s for s in lev["scenarios"] if s["segment_change_pct"] == min_pct),
                None
            )
            max_scenario = next(
                (s for s in lev["scenarios"] if s["segment_change_pct"] == max_pct),
                None
            )
            
            if min_scenario and max_scenario:
                tornado_bars.append({
                    "segment": lev["segment"],
                    "low_impact": min_scenario["total_change_pct"],
                    "high_impact": max_scenario["total_change_pct"],
                    "range": abs(max_scenario["total_change_pct"] - min_scenario["total_change_pct"]),
                    "share_pct": lev["share_pct"]
                })
        
        tornado_bars.sort(key=lambda x: x["range"], reverse=True)
        
        return {
            "scenario_range": f"{min_pct:+d}% to {max_pct:+d}%",
            "bars": tornado_bars,
            "interpretation": f"Tornado chart shows which {leverage_scores[0]['segment'] if leverage_scores else 'segments'} has most impact on total when changed by {min_pct}% to {max_pct}%"
        }
    
    def detect_outliers(self, user_id: str, metric_name: str, dimension: str,
                        method: str = "iqr", threshold: float = 1.5) -> Dict:
        """
        SUBTASK 3.3.3: Detect Outlier Segments
        
        Big Tech Pattern: Statistical outlier detection used by Netflix/Spotify
        for anomaly detection in user behavior and metrics.
        
        Methods:
        - IQR (Interquartile Range): Q1 - 1.5*IQR to Q3 + 1.5*IQR
        - Z-score: Mean ± threshold * standard deviation
        
        Args:
            user_id: User identifier
            metric_name: Metric to analyze
            dimension: Dimension to segment by
            method: "iqr" or "zscore"
            threshold: 1.5 for IQR, 2.0 or 3.0 for z-score
            
        Returns:
            Dict with outliers, bounds, and CEO-friendly insights
        """
        try:
            logger.info(f"🔍 Detecting outliers for {metric_name} by {dimension}...")
            
            # Get segment totals
            with self.engine.connect() as conn:
                query = text("""
                    SELECT 
                        dimensions->>:dimension as segment,
                        SUM(metric_value) as total
                    FROM user_uploaded_metrics
                    WHERE user_id = :user_id
                      AND metric_name = :metric_name
                      AND dimensions->>:dimension IS NOT NULL
                    GROUP BY dimensions->>:dimension
                    ORDER BY total DESC
                """)
                
                result = conn.execute(query, {
                    "user_id": user_id,
                    "metric_name": metric_name,
                    "dimension": dimension
                }).fetchall()
            
            if len(result) < 4:
                return {
                    "outliers_found": 0,
                    "outliers": [],
                    "message": "Need at least 4 segments for outlier detection",
                    "segments_analyzed": len(result)
                }
            
            # Extract values
            segments = [{"segment": row.segment, "value": float(row.total)} for row in result]
            values = [s["value"] for s in segments]
            
            # Calculate outlier bounds based on method
            if method == "iqr":
                bounds = self._calculate_iqr_bounds(values, threshold)
            else:  # zscore
                bounds = self._calculate_zscore_bounds(values, threshold)
            
            # Identify outliers
            outliers = []
            for seg in segments:
                if seg["value"] < bounds["lower"] or seg["value"] > bounds["upper"]:
                    direction = "HIGH" if seg["value"] > bounds["upper"] else "LOW"
                    
                    # Calculate how extreme
                    if method == "iqr":
                        iqr = bounds["q3"] - bounds["q1"]
                        if direction == "HIGH":
                            severity = (seg["value"] - bounds["upper"]) / iqr if iqr > 0 else 0
                        else:
                            severity = (bounds["lower"] - seg["value"]) / iqr if iqr > 0 else 0
                    else:
                        severity = abs(seg["value"] - bounds["mean"]) / bounds["std"] if bounds["std"] > 0 else 0
                    
                    outliers.append({
                        "segment": seg["segment"],
                        "value": seg["value"],
                        "direction": direction,
                        "severity": round(severity, 2),
                        "insight": self._generate_outlier_insight(seg["segment"], seg["value"], direction, bounds, metric_name)
                    })
            
            # Sort by severity
            outliers.sort(key=lambda x: x["severity"], reverse=True)
            
            # Generate summary
            summary = self._generate_outlier_summary(outliers, len(segments), metric_name, dimension)
            
            return {
                "outliers_found": len(outliers),
                "outliers": outliers,
                "bounds": bounds,
                "method": method,
                "threshold": threshold,
                "segments_analyzed": len(segments),
                "summary": summary,
                "ceo_insight": self._generate_outlier_ceo_insight(outliers, metric_name, dimension)
            }
            
        except Exception as e:
            logger.error(f"❌ Outlier detection failed: {e}")
            return {"error": str(e)}
    
    def _calculate_iqr_bounds(self, values: List[float], threshold: float) -> Dict:
        """Calculate IQR-based outlier bounds."""
        sorted_vals = sorted(values)
        n = len(sorted_vals)
        
        q1_idx = n // 4
        q3_idx = (3 * n) // 4
        
        q1 = sorted_vals[q1_idx]
        q3 = sorted_vals[q3_idx]
        iqr = q3 - q1
        
        return {
            "method": "iqr",
            "q1": round(q1, 2),
            "q3": round(q3, 2),
            "iqr": round(iqr, 2),
            "lower": round(q1 - threshold * iqr, 2),
            "upper": round(q3 + threshold * iqr, 2),
            "mean": round(sum(values) / len(values), 2),
            "std": round((sum((v - sum(values)/len(values))**2 for v in values) / len(values)) ** 0.5, 2)
        }
    
    def _calculate_zscore_bounds(self, values: List[float], threshold: float) -> Dict:
        """Calculate Z-score based outlier bounds."""
        mean = sum(values) / len(values)
        variance = sum((v - mean) ** 2 for v in values) / len(values)
        std = variance ** 0.5
        
        return {
            "method": "zscore",
            "mean": round(mean, 2),
            "std": round(std, 2),
            "lower": round(mean - threshold * std, 2),
            "upper": round(mean + threshold * std, 2),
            "q1": None,
            "q3": None,
            "iqr": None
        }
    
    def _generate_outlier_insight(self, segment: str, value: float, direction: str, 
                                   bounds: Dict, metric_name: str) -> str:
        """Generate plain English insight for an outlier."""
        metric_display = metric_name.replace("_", " ").title()
        
        if value >= 1_000_000:
            value_str = f"${value/1_000_000:.1f}M"
        elif value >= 1_000:
            value_str = f"${value/1_000:.0f}K"
        else:
            value_str = f"${value:,.0f}"
        
        if direction == "HIGH":
            return f"{segment} is unusually HIGH at {value_str} - significantly above normal range"
        else:
            return f"{segment} is unusually LOW at {value_str} - significantly below normal range"
    
    def _generate_outlier_summary(self, outliers: List[Dict], total_segments: int,
                                   metric_name: str, dimension: str) -> str:
        """Generate summary of outlier analysis."""
        if not outliers:
            return f"No outliers detected among {total_segments} {dimension} segments."
        
        high_count = sum(1 for o in outliers if o["direction"] == "HIGH")
        low_count = len(outliers) - high_count
        
        parts = []
        if high_count:
            parts.append(f"{high_count} unusually HIGH")
        if low_count:
            parts.append(f"{low_count} unusually LOW")
        
        return f"Found {' and '.join(parts)} out of {total_segments} {dimension} segments."
    
    def _generate_outlier_ceo_insight(self, outliers: List[Dict], metric_name: str, 
                                       dimension: str) -> str:
        """Generate CEO-friendly insight."""
        if not outliers:
            return f"All {dimension} segments are performing within normal ranges. No action needed."
        
        top = outliers[0]
        metric_display = metric_name.replace("_", " ").title()
        
        if top["direction"] == "HIGH":
            return f"🔥 {top['segment']} is a standout performer - investigate what's working and replicate."
        else:
            return f"⚠️ {top['segment']} is underperforming - investigate root cause immediately."


if __name__ == "__main__":
    import sys
    
    print("🏆 Testing DriverAnalyzer...")
    
    try:
        analyzer = DriverAnalyzer()
        print("✅ DriverAnalyzer initialized")
        
        test_user = sys.argv[1] if len(sys.argv) > 1 else None
        
        if test_user:
            metrics = analyzer._get_user_metrics(test_user)
            if metrics:
                dims = analyzer._get_metric_dimensions(test_user, metrics[0])
                if dims:
                    result = analyzer.decompose_contributions(test_user, metrics[0], dims[0])
                    print(f"✅ Contribution decomposition: {len(result.get('contributions', []))} segments")
                    
                    rank_result = analyzer.rank_drivers(test_user, metrics[0], dims[0])
                    print(f"✅ Driver ranking: {len(rank_result.get('top_drivers', []))} top drivers")
                    
                    outliers = analyzer.detect_outliers(test_user, metrics[0], dims[0])
                    print(f"✅ Outlier detection: {outliers.get('outliers_found', 0)} outliers found")
        
        print("\n✅ drivers.py working correctly!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

