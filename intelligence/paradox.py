"""
═══════════════════════════════════════════════════════════════════════════════
🧠 INTELLIGENCE ENGINE - Paradox Detection Module
═══════════════════════════════════════════════════════════════════════════════

Task 3.2: Simpson's Paradox Detection
- 3.2.1: Simpson's Paradox Scanner
- 3.2.2: Mix Shift Analyzer (Rate/Mix Decomposition)
- 3.2.3: Paradox Explainer (LLM-Powered)

Big Tech Pattern: Finds hidden contradictions in aggregated data.
"""

import json
import logging
from typing import Dict, List, Optional

from sqlalchemy import text

from .base import IntelligenceBase

logger = logging.getLogger(__name__)


class ParadoxDetector(IntelligenceBase):
    """
    🎭 Simpson's Paradox Detector
    
    THE MONSTER FEATURE: Finds when overall trend contradicts segment trends.
    This is what separates amateur analysis from PhD-level insights.
    """
    
    # ═══════════════════════════════════════════════════════════════════════════
    # 🎭 SUBTASK 3.2.1: SIMPSON'S PARADOX SCANNER
    # ═══════════════════════════════════════════════════════════════════════════
    
    def scan_simpsons_paradox(self, user_id: str, metric_name: str,
                               dimension: str) -> Dict:
        """
        SUBTASK 3.2.1: Detect Simpson's Paradox in data.
        
        Simpson's Paradox: A trend appears in combined data but disappears or
        REVERSES when data is split into groups.
        """
        try:
            overall_growth = self._calculate_period_growth(user_id, metric_name)
            
            if overall_growth is None:
                return {"error": "Could not calculate overall growth"}
            
            dim_values = self._get_dimension_unique_values(user_id, metric_name, dimension)
            
            if len(dim_values) < 2:
                return {
                    "error": f"Need at least 2 values in dimension '{dimension}', found {len(dim_values)}",
                    "dimension_values": dim_values
                }
            
            segment_trends = {}
            segment_shares = {}
            
            for value in dim_values:
                growth = self._calculate_period_growth(user_id, metric_name, dimension, value)
                share_change = self._calculate_share_change(user_id, metric_name, dimension, value)
                
                if growth is not None:
                    segment_trends[value] = growth
                if share_change is not None:
                    segment_shares[value] = share_change
            
            if len(segment_trends) < 2:
                return {"error": "Could not calculate enough segment trends"}
            
            paradox_result = self._detect_paradox_type(overall_growth, segment_trends)
            
            mix_shift_analysis = None
            if paradox_result["paradox_detected"]:
                mix_shift_analysis = self._analyze_mix_shift(segment_trends, segment_shares)
            
            explanation = self._explain_paradox(
                paradox_result, overall_growth, segment_trends,
                segment_shares, dimension, mix_shift_analysis
            )
            
            result = {
                "metric_name": metric_name,
                "dimension": dimension,
                "paradox_detected": paradox_result["paradox_detected"],
                "paradox_type": paradox_result.get("paradox_type"),
                "severity": paradox_result.get("severity"),
                "overall_trend": {
                    "growth_pct": round(overall_growth, 2),
                    "direction": "up" if overall_growth > 0 else "down" if overall_growth < 0 else "flat"
                },
                "segment_trends": {
                    k: {
                        "growth_pct": round(v, 2),
                        "direction": "up" if v > 0 else "down" if v < 0 else "flat",
                        "share_change_pct": round(segment_shares.get(k, 0), 2)
                    }
                    for k, v in segment_trends.items()
                },
                "conflicting_segments": paradox_result.get("conflicting_segments", []),
                "mix_shift_analysis": mix_shift_analysis,
                "explanation": explanation,
                "insight_level": "🔥 CRITICAL" if paradox_result["paradox_detected"] else "✅ OK"
            }
            
            if paradox_result["paradox_detected"]:
                logger.warning(f"🎭 SIMPSON'S PARADOX DETECTED for user {user_id[:8]}...: {dimension}")
            else:
                logger.info(f"🎭 No paradox found for user {user_id[:8]}...: {dimension}")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Simpson's Paradox scan failed: {e}")
            return {"error": str(e)}

    # ═══════════════════════════════════════════════════════════════════════════
    # 🚀 GTC 2026: UNIVERSAL PARADOX ANALYSIS (JSONB Data + NVIDIA Narrative)
    # ═══════════════════════════════════════════════════════════════════════════

    def analyze_uploaded_data(self, user_id: str, metric_name: str,
                               dimension: str) -> Dict:
        """
        🚀 GTC 2026: Simpson's Paradox for generic CSV data (JSONB storage).

        Universal: Works with ANY uploaded CSV without date columns.
        Pattern: Big Tech - analyze aggregates vs segments to find hidden paradoxes.
        NVIDIA: Uses Nemotron for CEO-level narrative generation.
        """
        try:
            logger.info(f"🎭 UNIVERSAL Paradox: metric='{metric_name}', dimension='{dimension}'")

            # Step 1: Get schema to find available columns
            from tools.sql_tools import get_user_schema
            schema = get_user_schema(user_id)

            if not schema:
                return {"error": "No schema found for user", "method": "analyze_uploaded_data"}

            all_columns = schema.get('all_columns', [])

            # Guard: Auto-detect metric and dimension if None
            if not metric_name or not dimension:
                _numeric_hints = ['salary', 'revenue', 'income', 'sales', 'amount', 'price', 'age', 'spending']
                _categorical_hints = ['education', 'gender', 'department', 'region', 'category', 'status', 'marital']
                if not metric_name:
                    for col in all_columns:
                        if any(h in col.lower() for h in _numeric_hints):
                            metric_name = col
                            break
                if not dimension:
                    for col in all_columns:
                        if any(h in col.lower() for h in _categorical_hints):
                            dimension = col
                            break
                if not metric_name or not dimension:
                    return {"error": f"Could not determine metric or dimension from columns: {all_columns[:10]}", "method": "analyze_uploaded_data"}
                logger.info(f"🎭 Auto-detected: metric='{metric_name}', dimension='{dimension}'")

            # Step 3: Query overall averages by dimension
            # Strategy: Try JSONB first (user_uploaded_rows), fallback to metrics table
            overall_by_dim = self._query_jsonb_aggregate(
                user_id, metric_name, dimension
            )

            # Step 3b: Fallback to metrics table (user_uploaded_metrics - lowercase keys)
            using_metrics_table = False
            if not overall_by_dim or len(overall_by_dim) < 2:
                logger.info(f"🎭 JSONB returned {len(overall_by_dim) if overall_by_dim else 0} groups, trying metrics table")
                metric_lower = metric_name.lower()
                dimension_lower = dimension.lower()
                overall_by_dim = self._query_metrics_aggregate(
                    user_id, metric_lower, dimension_lower
                )
                if overall_by_dim and len(overall_by_dim) >= 2:
                    using_metrics_table = True
                    logger.info(f"🎭 Metrics table SUCCESS: {len(overall_by_dim)} groups for '{dimension}'")

            if not overall_by_dim or len(overall_by_dim) < 2:
                return {
                    "error": f"Need at least 2 groups in '{dimension}' (tried both JSONB and metrics tables)",
                    "method": "analyze_uploaded_data"
                }

            # Step 4: Detect secondary dimension and query sub-group averages
            subgroup_data = {}
            secondary_dimension = None

            if using_metrics_table:
                # Discover dimension keys from metrics table JSONB
                metric_lower = metric_name.lower()
                dimension_lower = dimension.lower()
                dimension_keys = self._get_metrics_dimension_keys(user_id, metric_lower)
                for key in dimension_keys:
                    if key != dimension_lower:
                        secondary_dimension = key
                        logger.info(f"🎭 Metrics secondary dimension: '{key}'")
                        break
                if secondary_dimension:
                    subgroup_data = self._query_metrics_subgroups(
                        user_id, metric_lower, dimension_lower, secondary_dimension
                    )
            else:
                # JSONB path: detect from schema columns (multi-file safe)
                secondary_dimension = self._detect_secondary_dimension(
                    all_columns, metric_name, dimension, user_id=user_id
                )
                if secondary_dimension:
                    subgroup_data = self._query_jsonb_subgroups(
                        user_id, metric_name, dimension, secondary_dimension
                    )

            # Step 5: Detect paradox
            paradox_result = self._detect_jsonb_paradox(
                overall_by_dim, subgroup_data, dimension, secondary_dimension
            )

            # 🚀 GTC 2026: Generate NVIDIA-powered narrative
            nvidia_narrative = self._generate_nvidia_paradox_narrative(
                overall_by_dim, subgroup_data, paradox_result,
                metric_name, dimension, secondary_dimension
            )

            # Step 6: Build response
            result = {
                "metric_name": metric_name,
                "dimension": dimension,
                "secondary_dimension": secondary_dimension,
                "paradox_detected": paradox_result.get("paradox_detected", False),
                "paradox_type": paradox_result.get("paradox_type"),
                "severity": paradox_result.get("severity", "low"),
                "overall_by_dimension": overall_by_dim,
                "subgroup_analysis": subgroup_data if subgroup_data else None,
                "explanation": nvidia_narrative or paradox_result.get("explanation", ""),
                "insight_level": "🔥 CRITICAL" if paradox_result.get("paradox_detected") else "✅ OK",
                "method": "analyze_uploaded_data",
                "storage_type": "metrics" if using_metrics_table else "jsonb",
                "nvidia_powered": True
            }

            if paradox_result.get("paradox_detected"):
                logger.warning(f"🎭 SIMPSON'S PARADOX DETECTED (UNIVERSAL): {dimension}")
            else:
                logger.info(f"🎭 No paradox found (UNIVERSAL): {dimension}")

            return result

        except Exception as e:
            logger.error(f"❌ Universal Paradox analysis failed: {e}")
            return {"error": str(e), "method": "analyze_uploaded_data"}

    def _generate_nvidia_paradox_narrative(self, overall: Dict, subgroups: Dict,
                                            paradox_result: Dict, metric_name: str,
                                            dimension: str, secondary_dimension: Optional[str]) -> Optional[str]:
        """
        🚀 GTC 2026: Generate CEO-level narrative using NVIDIA Nemotron.

        Pattern: Google/OpenAI - LLM-powered insight generation
        Universal: Works with any data, any paradox result
        """
        try:
            from tools.llm_router_client import LLMRouterClient

            # Build context for LLM (with safety checks)
            overall_parts = []
            for k, v in overall.items():
                if isinstance(v, dict) and 'avg' in v:
                    overall_parts.append(f"{k}: avg {v['avg']:,.0f} (n={v.get('count', 0)})")
                else:
                    overall_parts.append(f"{k}: {v}")
            overall_summary = ", ".join(overall_parts)

            subgroup_summary = ""
            if subgroups and secondary_dimension:
                for sub_name, sub_data in list(subgroups.items())[:3]:  # Limit to 3
                    if isinstance(sub_data, dict):
                        sub_parts = []
                        for k, v in sub_data.items():
                            if isinstance(v, dict) and 'avg' in v:
                                sub_parts.append(f"{k}: {v['avg']:,.0f}")
                            else:
                                sub_parts.append(f"{k}: {v}")
                        sub_details = ", ".join(sub_parts)
                    else:
                        sub_details = str(sub_data)
                    subgroup_summary += f"  {sub_name}: {sub_details}\n"

            paradox_detected = paradox_result.get("paradox_detected", False)

            prompt = f"""You are a senior data analyst. Generate a clear, actionable insight about this data analysis.

ANALYSIS TYPE: Simpson's Paradox Detection
METRIC: {metric_name}
PRIMARY DIMENSION: {dimension}
SECONDARY DIMENSION: {secondary_dimension or "None"}

OVERALL BY {dimension.upper()}:
{overall_summary}

{"BREAKDOWN BY " + secondary_dimension.upper() + ":" if secondary_dimension else ""}
{subgroup_summary}

PARADOX DETECTED: {"YES - the overall trend REVERSES in subgroups!" if paradox_detected else "NO - trends are consistent"}

INSTRUCTIONS:
- Write 2-3 sentences as a CEO-level insight
- If paradox detected: Explain WHY the aggregate is misleading and what action to take
- If no paradox: Confirm the data is consistent and highlight the key finding
- Be specific with numbers
- End with a clear recommendation

INSIGHT:"""

            # 🚀 GTC 2026: Use NVIDIA Router (Tier 2 = Ultra 253B for PhD-level analysis)
            router = LLMRouterClient(tier=2)
            result = router.call(prompt)

            # Big Tech Pattern: ChatCompletion uses choices[0].message.content
            if result and hasattr(result, 'choices') and result.choices:
                narrative = str(result.choices[0].message.content).strip()
                logger.info(f"🚀 NVIDIA Router: Paradox narrative generated ({len(narrative)} chars)")
                return narrative

        except Exception as e:
            logger.warning(f"⚠️ NVIDIA narrative failed, using template: {e}")

        return None

    def _detect_secondary_dimension(self, all_columns: List[str],
                                     metric_name: str, primary_dimension: str,
                                     user_id: str = None) -> Optional[str]:
        """Auto-detect a secondary dimension for sub-group analysis.

        Multi-file safe: Only picks columns that co-exist with the metric
        in the same rows (same file). Prevents cross-file column confusion
        when multiple files are loaded in user_uploaded_rows.
        """
        # Multi-file fix: get columns from rows that actually have the metric
        file_columns = set(all_columns)
        if user_id and self.engine:
            try:
                with self.engine.connect() as conn:
                    result = conn.execute(text("""
                        SELECT DISTINCT jsonb_object_keys(row_data)
                        FROM (
                            SELECT row_data FROM user_uploaded_rows
                            WHERE user_id = :uid
                              AND row_data->>:metric IS NOT NULL
                            LIMIT 1
                        ) sub
                    """), {"uid": user_id, "metric": metric_name})
                    file_columns = set(r[0] for r in result.fetchall())
                    logger.info(f"🎭 Multi-file filter: {len(file_columns)} cols from file with '{metric_name}'")
            except Exception as e:
                logger.warning(f"⚠️ Multi-file column filter failed: {e}")

        categorical_hints = [
            'department', 'region', 'category', 'type', 'status', 'level',
            'gender', 'country', 'city', 'team', 'group', 'segment',
            'job', 'title', 'role', 'class', 'tier'
        ]

        for col in all_columns:
            if col not in file_columns:
                continue
            col_lower = col.lower()
            if col_lower == metric_name.lower() or col_lower == primary_dimension.lower():
                continue
            if any(skip in col_lower for skip in ['id', 'name', 'email', 'phone', 'date', 'time']):
                continue
            if any(hint in col_lower for hint in categorical_hints):
                logger.info(f"🎭 Auto-detected secondary dimension: '{col}'")
                return col

        return None

    def _query_jsonb_aggregate(self, user_id: str, metric_name: str,
                                dimension: str) -> Dict[str, Dict]:
        """Query JSONB data for aggregate statistics by dimension."""
        try:
            with self.engine.connect() as conn:
                query = text("""
                    SELECT
                        row_data->>:dimension AS dim_value,
                        AVG((row_data->>:metric)::numeric) AS avg_value,
                        COUNT(*) AS count
                    FROM user_uploaded_rows
                    WHERE user_id = :user_id
                      AND row_data->>:dimension IS NOT NULL
                      AND row_data->>:metric IS NOT NULL
                    GROUP BY row_data->>:dimension
                    ORDER BY avg_value DESC
                """)

                result = conn.execute(query, {
                    "user_id": user_id,
                    "dimension": dimension,
                    "metric": metric_name
                })

                data = {}
                for row in result:
                    dim_val = row[0]
                    if dim_val:
                        data[dim_val] = {
                            "avg": round(float(row[1]), 2) if row[1] else 0,
                            "count": int(row[2]) if row[2] else 0
                        }

                logger.info(f"🎭 JSONB aggregate: {len(data)} groups for '{dimension}'")
                return data

        except Exception as e:
            logger.error(f"❌ JSONB aggregate query failed: {e}")
            return {}

    def _query_jsonb_subgroups(self, user_id: str, metric_name: str,
                                dimension: str, secondary_dimension: str) -> Dict[str, Dict]:
        """Query JSONB data for sub-group analysis."""
        try:
            with self.engine.connect() as conn:
                query = text("""
                    SELECT
                        row_data->>:secondary AS secondary_val,
                        row_data->>:dimension AS dim_value,
                        AVG((row_data->>:metric)::numeric) AS avg_value,
                        COUNT(*) AS count
                    FROM user_uploaded_rows
                    WHERE user_id = :user_id
                      AND row_data->>:dimension IS NOT NULL
                      AND row_data->>:secondary IS NOT NULL
                      AND row_data->>:metric IS NOT NULL
                    GROUP BY row_data->>:secondary, row_data->>:dimension
                    ORDER BY row_data->>:secondary, avg_value DESC
                """)

                result = conn.execute(query, {
                    "user_id": user_id,
                    "dimension": dimension,
                    "secondary": secondary_dimension,
                    "metric": metric_name
                })

                data = {}
                for row in result:
                    secondary_val = row[0]
                    dim_val = row[1]

                    if secondary_val and dim_val:
                        if secondary_val not in data:
                            data[secondary_val] = {}
                        data[secondary_val][dim_val] = {
                            "avg": round(float(row[2]), 2) if row[2] else 0,
                            "count": int(row[3]) if row[3] else 0
                        }

                logger.info(f"🎭 JSONB subgroups: {len(data)} secondary groups")
                return data

        except Exception as e:
            logger.error(f"❌ JSONB subgroup query failed: {e}")
            return {}

    def _detect_jsonb_paradox(self, overall: Dict, subgroups: Dict,
                               dimension: str, secondary_dimension: Optional[str]) -> Dict:
        """Detect Simpson's Paradox in JSONB data."""
        try:
            groups = list(overall.keys())
            if len(groups) < 2:
                return {"paradox_detected": False, "explanation": "Need at least 2 groups"}

            sorted_groups = sorted(groups, key=lambda g: overall[g]["avg"], reverse=True)
            winner_overall = sorted_groups[0]
            loser_overall = sorted_groups[1]

            if not subgroups or not secondary_dimension:
                return {
                    "paradox_detected": False,
                    "explanation": f"{winner_overall} leads overall. No secondary dimension for sub-analysis."
                }

            reversals = []
            for sub_name, sub_data in subgroups.items():
                if winner_overall in sub_data and loser_overall in sub_data:
                    winner_avg = sub_data[winner_overall]["avg"]
                    loser_avg = sub_data[loser_overall]["avg"]

                    if loser_avg > winner_avg:
                        reversals.append({
                            "subgroup": sub_name,
                            "winner_in_subgroup": loser_overall,
                            "winner_avg": loser_avg,
                            "loser_avg": winner_avg
                        })

            paradox_detected = len(reversals) > 0

            if paradox_detected:
                severity = "high" if len(reversals) >= len(subgroups) / 2 else "medium"
                return {
                    "paradox_detected": True,
                    "paradox_type": "simpsons_reversal",
                    "severity": severity,
                    "reversals": reversals,
                    "explanation": f"Paradox: {winner_overall} leads overall but {loser_overall} leads in {len(reversals)} subgroups"
                }
            else:
                return {
                    "paradox_detected": False,
                    "explanation": f"No paradox: {winner_overall} leads consistently"
                }

        except Exception as e:
            logger.error(f"❌ Paradox detection failed: {e}")
            return {"paradox_detected": False, "error": str(e)}

    # ═══════════════════════════════════════════════════════════════════════════
    # 🚀 GTC 2026: METRICS TABLE QUERIES (user_uploaded_metrics - lowercase keys)
    # ═══════════════════════════════════════════════════════════════════════════
    # Same return format as JSONB methods so _detect_jsonb_paradox() works with either.
    # Pattern: Strategy - same interface, different data source.
    # ═══════════════════════════════════════════════════════════════════════════

    def _query_metrics_aggregate(self, user_id: str, metric_name: str,
                                  dimension: str) -> Dict[str, Dict]:
        """
        Query user_uploaded_metrics for aggregate statistics by dimension.

        Same return format as _query_jsonb_aggregate():
            {dim_value: {"avg": float, "count": int}, ...}

        Metrics table stores lowercase keys. dimension is a key inside
        the JSONB 'dimensions' column (e.g., dimensions->>'education').
        """
        try:
            with self.engine.connect() as conn:
                query = text("""
                    SELECT
                        dimensions->>:dimension AS dim_value,
                        AVG(metric_value) AS avg_value,
                        COUNT(*) AS count
                    FROM user_uploaded_metrics
                    WHERE user_id = :user_id
                      AND metric_name = :metric
                      AND dimensions->>:dimension IS NOT NULL
                    GROUP BY dimensions->>:dimension
                    ORDER BY avg_value DESC
                """)

                result = conn.execute(query, {
                    "user_id": user_id,
                    "metric": metric_name,
                    "dimension": dimension
                })

                data = {}
                for row in result:
                    dim_val = row[0]
                    if dim_val:
                        data[dim_val] = {
                            "avg": round(float(row[1]), 2) if row[1] else 0,
                            "count": int(row[2]) if row[2] else 0
                        }

                logger.info(f"🎭 Metrics aggregate: {len(data)} groups for '{dimension}' (metric='{metric_name}')")
                return data

        except Exception as e:
            logger.error(f"❌ Metrics aggregate query failed: {e}")
            return {}

    def _query_metrics_subgroups(self, user_id: str, metric_name: str,
                                  dimension: str, secondary_dimension: str) -> Dict[str, Dict]:
        """
        Query user_uploaded_metrics for sub-group analysis.

        Same return format as _query_jsonb_subgroups():
            {secondary_val: {dim_val: {"avg": float, "count": int}, ...}, ...}
        """
        try:
            with self.engine.connect() as conn:
                query = text("""
                    SELECT
                        dimensions->>:secondary AS secondary_val,
                        dimensions->>:dimension AS dim_value,
                        AVG(metric_value) AS avg_value,
                        COUNT(*) AS count
                    FROM user_uploaded_metrics
                    WHERE user_id = :user_id
                      AND metric_name = :metric
                      AND dimensions->>:dimension IS NOT NULL
                      AND dimensions->>:secondary IS NOT NULL
                    GROUP BY dimensions->>:secondary, dimensions->>:dimension
                    ORDER BY dimensions->>:secondary, avg_value DESC
                """)

                result = conn.execute(query, {
                    "user_id": user_id,
                    "metric": metric_name,
                    "dimension": dimension,
                    "secondary": secondary_dimension
                })

                data = {}
                for row in result:
                    secondary_val = row[0]
                    dim_val = row[1]

                    if secondary_val and dim_val:
                        if secondary_val not in data:
                            data[secondary_val] = {}
                        data[secondary_val][dim_val] = {
                            "avg": round(float(row[2]), 2) if row[2] else 0,
                            "count": int(row[3]) if row[3] else 0
                        }

                logger.info(f"🎭 Metrics subgroups: {len(data)} secondary groups")
                return data

        except Exception as e:
            logger.error(f"❌ Metrics subgroup query failed: {e}")
            return {}

    def _get_metrics_dimension_keys(self, user_id: str, metric_name: str) -> List[str]:
        """
        Discover available dimension keys from the metrics table JSONB dimensions column.

        Returns list of dimension key names (e.g., ['education', 'marital_status']).
        Used to find secondary dimensions when data is in metrics table.
        """
        try:
            with self.engine.connect() as conn:
                query = text("""
                    SELECT DISTINCT k
                    FROM user_uploaded_metrics
                    CROSS JOIN LATERAL jsonb_object_keys(dimensions) AS k
                    WHERE user_id = :user_id
                      AND metric_name = :metric
                """)

                result = conn.execute(query, {
                    "user_id": user_id,
                    "metric": metric_name
                })

                keys = [row[0] for row in result if row[0]]
                logger.info(f"🎭 Metrics dimension keys for '{metric_name}': {keys}")
                return keys

        except Exception as e:
            logger.error(f"❌ Metrics dimension keys query failed: {e}")
            return []

    def _calculate_period_growth(self, user_id: str, metric_name: str,
                                  dimension: Optional[str] = None,
                                  dim_value: Optional[str] = None) -> Optional[float]:
        """Calculate growth rate between first half and second half of data."""
        try:
            with self.engine.connect() as conn:
                dim_filter = ""
                params = {"user_id": user_id, "metric_name": metric_name}
                
                if dimension and dim_value:
                    dim_filter = "AND dimensions->>:dimension = :dim_value"
                    params["dimension"] = dimension
                    params["dim_value"] = dim_value
                
                query = text(f"""
                    WITH date_range AS (
                        SELECT 
                            MIN(date) as min_date,
                            MAX(date) as max_date,
                            (MAX(date) - MIN(date)) / 2 as mid_offset
                        FROM user_uploaded_metrics
                        WHERE user_id = :user_id
                          AND metric_name = :metric_name
                        {dim_filter}
                    ),
                    first_half AS (
                        SELECT SUM(metric_value) as total
                        FROM user_uploaded_metrics, date_range
                        WHERE user_id = :user_id
                          AND metric_name = :metric_name
                          AND date < date_range.min_date + date_range.mid_offset
                        {dim_filter}
                    ),
                    second_half AS (
                        SELECT SUM(metric_value) as total
                        FROM user_uploaded_metrics, date_range
                        WHERE user_id = :user_id
                          AND metric_name = :metric_name
                          AND date >= date_range.min_date + date_range.mid_offset
                        {dim_filter}
                    )
                    SELECT first_half.total as first_total, second_half.total as second_total
                    FROM first_half, second_half
                """)
                
                result = conn.execute(query, params).fetchone()
                
                if result and result.first_total and result.second_total:
                    first = float(result.first_total)
                    second = float(result.second_total)
                    
                    if first != 0:
                        return ((second - first) / first) * 100
                
                return None
                
        except Exception as e:
            logger.warning(f"⚠️ Period growth calculation failed: {e}")
            return None
    
    def _calculate_share_change(self, user_id: str, metric_name: str,
                                 dimension: str, dim_value: str) -> Optional[float]:
        """Calculate how a segment's share of total changed over time."""
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
                    first_half_segment AS (
                        SELECT COALESCE(SUM(metric_value), 0) as seg_total
                        FROM user_uploaded_metrics, date_range
                        WHERE user_id = :user_id
                          AND metric_name = :metric_name
                          AND dimensions->>:dimension = :dim_value
                          AND date < date_range.min_date + date_range.mid_offset
                    ),
                    first_half_total AS (
                        SELECT COALESCE(SUM(metric_value), 0) as all_total
                        FROM user_uploaded_metrics, date_range
                        WHERE user_id = :user_id
                          AND metric_name = :metric_name
                          AND date < date_range.min_date + date_range.mid_offset
                    ),
                    second_half_segment AS (
                        SELECT COALESCE(SUM(metric_value), 0) as seg_total
                        FROM user_uploaded_metrics, date_range
                        WHERE user_id = :user_id
                          AND metric_name = :metric_name
                          AND dimensions->>:dimension = :dim_value
                          AND date >= date_range.min_date + date_range.mid_offset
                    ),
                    second_half_total AS (
                        SELECT COALESCE(SUM(metric_value), 0) as all_total
                        FROM user_uploaded_metrics, date_range
                        WHERE user_id = :user_id
                          AND metric_name = :metric_name
                          AND date >= date_range.min_date + date_range.mid_offset
                    )
                    SELECT 
                        first_half_segment.seg_total as first_seg,
                        first_half_total.all_total as first_all,
                        second_half_segment.seg_total as second_seg,
                        second_half_total.all_total as second_all
                    FROM first_half_segment, first_half_total, 
                         second_half_segment, second_half_total
                """)
                
                result = conn.execute(query, {
                    "user_id": user_id,
                    "metric_name": metric_name,
                    "dimension": dimension,
                    "dim_value": dim_value
                }).fetchone()
                
                if result:
                    first_share = (float(result.first_seg) / float(result.first_all) * 100) if result.first_all else 0
                    second_share = (float(result.second_seg) / float(result.second_all) * 100) if result.second_all else 0
                    return second_share - first_share
                
                return None
                
        except Exception as e:
            logger.warning(f"⚠️ Share change calculation failed: {e}")
            return None
    
    def _detect_paradox_type(self, overall_growth: float,
                              segment_trends: Dict[str, float]) -> Dict:
        """Detect if Simpson's Paradox exists and classify its type."""
        overall_direction = 1 if overall_growth > 0 else -1 if overall_growth < 0 else 0
        
        same_direction = 0
        opposite_direction = 0
        conflicting = []
        
        for segment, growth in segment_trends.items():
            seg_direction = 1 if growth > 0 else -1 if growth < 0 else 0
            
            if seg_direction == overall_direction:
                same_direction += 1
            elif seg_direction == -overall_direction and overall_direction != 0:
                opposite_direction += 1
                conflicting.append(segment)
        
        total_segments = len(segment_trends)
        
        if opposite_direction > total_segments / 2:
            severity = "severe" if opposite_direction == total_segments else "moderate"
            return {
                "paradox_detected": True,
                "paradox_type": "reversal",
                "severity": severity,
                "conflicting_segments": conflicting,
                "opposite_count": opposite_direction,
                "total_segments": total_segments
            }
        
        if same_direction == total_segments and overall_direction != 0:
            avg_segment_growth = sum(segment_trends.values()) / len(segment_trends)
            
            if abs(overall_growth) > abs(avg_segment_growth) * 2:
                return {
                    "paradox_detected": True,
                    "paradox_type": "attenuation",
                    "severity": "moderate",
                    "conflicting_segments": [],
                    "note": "Overall growth much stronger than segment average"
                }
        
        return {
            "paradox_detected": False,
            "paradox_type": None,
            "severity": None,
            "conflicting_segments": []
        }
    
    def _analyze_mix_shift(self, segment_trends: Dict[str, float],
                            segment_shares: Dict[str, float]) -> Dict:
        """Analyze the mix shift causing the paradox."""
        growing_share = []
        shrinking_share = []
        
        for segment, share_change in segment_shares.items():
            trend = segment_trends.get(segment, 0)
            
            if share_change > 1:
                growing_share.append({
                    "segment": segment,
                    "share_change": round(share_change, 2),
                    "growth": round(trend, 2)
                })
            elif share_change < -1:
                shrinking_share.append({
                    "segment": segment,
                    "share_change": round(share_change, 2),
                    "growth": round(trend, 2)
                })
        
        growing_share.sort(key=lambda x: -x["share_change"])
        shrinking_share.sort(key=lambda x: x["share_change"])
        
        return {
            "segments_gaining_share": growing_share,
            "segments_losing_share": shrinking_share,
            "primary_driver": growing_share[0]["segment"] if growing_share else None,
            "mix_shift_detected": len(growing_share) > 0 or len(shrinking_share) > 0
        }
    
    def _explain_paradox(self, paradox_result: Dict, overall_growth: float,
                          segment_trends: Dict[str, float],
                          segment_shares: Dict[str, float],
                          dimension: str, mix_shift: Optional[Dict]) -> str:
        """Generate human-readable explanation of the paradox."""
        if not paradox_result["paradox_detected"]:
            return f"No paradox detected. Overall trend ({overall_growth:+.1f}%) is consistent with {dimension} segment trends."
        
        overall_dir = "up" if overall_growth > 0 else "down"
        
        if paradox_result["paradox_type"] == "reversal":
            conflicting = paradox_result["conflicting_segments"]
            
            seg_summary = ", ".join([
                f"{seg}: {segment_trends[seg]:+.1f}%"
                for seg in conflicting[:3]
            ])
            
            explanation = (
                f"🎭 SIMPSON'S PARADOX DETECTED!\n\n"
                f"Overall metric is {overall_dir} {abs(overall_growth):.1f}%, BUT:\n"
                f"• {len(conflicting)}/{len(segment_trends)} {dimension} segments show OPPOSITE trend\n"
                f"• Conflicting: {seg_summary}\n\n"
            )
            
            if mix_shift and mix_shift.get("primary_driver"):
                driver = mix_shift["primary_driver"]
                driver_share = segment_shares.get(driver, 0)
                driver_trend = segment_trends.get(driver, 0)
                
                explanation += (
                    f"ROOT CAUSE: Mix shift in {dimension} composition\n"
                    f"• {driver} gained {driver_share:+.1f}% share while growing {driver_trend:+.1f}%\n"
                    f"• The 'overall growth' is composition change, NOT real improvement!\n\n"
                    f"⚠️ ACTION: Investigate each {dimension} segment separately."
                )
            
            return explanation
        
        elif paradox_result["paradox_type"] == "attenuation":
            avg_seg = sum(segment_trends.values()) / len(segment_trends)
            
            return (
                f"🎭 SIMPSON'S PARADOX (Attenuation)!\n\n"
                f"Overall growth ({overall_growth:+.1f}%) is much stronger than "
                f"average segment growth ({avg_seg:+.1f}%).\n\n"
                f"This suggests mix shift: high-performing segments are gaining share, "
                f"artificially inflating the overall number.\n\n"
                f"⚠️ ACTION: Don't celebrate the {abs(overall_growth):.1f}% growth — "
                f"it may not reflect real improvement."
            )
        
        return "Paradox detected but could not generate explanation."
    
    # ═══════════════════════════════════════════════════════════════════════════
    # 📊 SUBTASK 3.2.2: MIX SHIFT ANALYZER (Rate/Mix Decomposition)
    # ═══════════════════════════════════════════════════════════════════════════
    
    def analyze_mix_shift_detailed(self, user_id: str, metric_name: str,
                                    dimension: str) -> Dict:
        """
        SUBTASK 3.2.2: PhD-level mix shift decomposition.
        
        Decomposes total change into:
        1. RATE EFFECT: Growth if composition stayed constant
        2. MIX EFFECT: Growth from changing composition
        3. INTERACTION EFFECT: Combined effect
        """
        try:
            segment_data = self._get_period_segment_data(user_id, metric_name, dimension)
            
            if not segment_data or len(segment_data) < 2:
                return {
                    "error": f"Need at least 2 segments, found {len(segment_data) if segment_data else 0}"
                }
            
            total_first = sum(s["first_value"] for s in segment_data.values())
            total_second = sum(s["second_value"] for s in segment_data.values())
            
            if total_first == 0:
                return {"error": "First period total is zero, cannot calculate change"}
            
            total_change = total_second - total_first
            total_change_pct = (total_change / total_first) * 100
            
            rate_effect = 0
            mix_effect = 0
            interaction_effect = 0
            segment_details = {}
            
            for segment, data in segment_data.items():
                first_val = data["first_value"]
                second_val = data["second_value"]
                
                first_share = first_val / total_first if total_first > 0 else 0
                second_share = second_val / total_second if total_second > 0 else 0
                
                seg_change = second_val - first_val
                
                if first_share > 0:
                    seg_growth_rate = (second_val / first_val - 1) if first_val > 0 else 0
                else:
                    seg_growth_rate = 0
                
                share_change = second_share - first_share
                
                seg_rate_effect = first_share * total_first * seg_growth_rate
                seg_mix_effect = share_change * total_first * (1 + seg_growth_rate / 2)
                seg_interaction = seg_change - seg_rate_effect - seg_mix_effect
                
                rate_effect += seg_rate_effect
                mix_effect += seg_mix_effect
                interaction_effect += seg_interaction
                
                segment_details[segment] = {
                    "first_period": {
                        "value": round(first_val, 2),
                        "share_pct": round(first_share * 100, 2)
                    },
                    "second_period": {
                        "value": round(second_val, 2),
                        "share_pct": round(second_share * 100, 2)
                    },
                    "change": {
                        "absolute": round(seg_change, 2),
                        "pct": round(seg_growth_rate * 100, 2) if first_val > 0 else None,
                        "share_change_pct": round(share_change * 100, 2)
                    },
                    "contribution_to_total_change_pct": round((seg_change / total_change * 100) if total_change != 0 else 0, 1)
                }
            
            if total_change != 0:
                rate_effect_pct = (rate_effect / abs(total_change)) * 100
                mix_effect_pct = (mix_effect / abs(total_change)) * 100
                interaction_pct = (interaction_effect / abs(total_change)) * 100
            else:
                rate_effect_pct = mix_effect_pct = interaction_pct = 0
            
            interpretation = self._interpret_mix_shift(
                total_change_pct, rate_effect_pct, mix_effect_pct,
                segment_details, dimension
            )
            
            result = {
                "metric_name": metric_name,
                "dimension": dimension,
                "total_change": {
                    "absolute": round(total_change, 2),
                    "pct": round(total_change_pct, 2)
                },
                "decomposition": {
                    "rate_effect": {
                        "absolute": round(rate_effect, 2),
                        "pct_of_change": round(rate_effect_pct, 1),
                        "meaning": "Change from segments growing/shrinking"
                    },
                    "mix_effect": {
                        "absolute": round(mix_effect, 2),
                        "pct_of_change": round(mix_effect_pct, 1),
                        "meaning": "Change from composition shift"
                    },
                    "interaction_effect": {
                        "absolute": round(interaction_effect, 2),
                        "pct_of_change": round(interaction_pct, 1),
                        "meaning": "Combined rate and mix effect"
                    }
                },
                "segment_details": segment_details,
                "top_contributors": self._get_top_contributors(segment_details),
                "interpretation": interpretation
            }
            
            logger.info(f"📊 Mix shift analysis for user {user_id[:8]}...: rate={rate_effect_pct:.1f}%, mix={mix_effect_pct:.1f}%")
            return result
            
        except Exception as e:
            logger.error(f"❌ Mix shift analysis failed: {e}")
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
    
    def _get_top_contributors(self, segment_details: Dict) -> List[Dict]:
        """Get top contributors to total change."""
        contributors = []
        
        for segment, data in segment_details.items():
            contribution = data.get("contribution_to_total_change_pct", 0)
            change = data.get("change", {})
            
            contributors.append({
                "segment": segment,
                "contribution_pct": contribution,
                "segment_growth_pct": change.get("pct"),
                "share_change_pct": change.get("share_change_pct", 0)
            })
        
        contributors.sort(key=lambda x: abs(x["contribution_pct"]), reverse=True)
        
        return contributors[:5]
    
    def _interpret_mix_shift(self, total_change_pct: float, rate_effect_pct: float,
                              mix_effect_pct: float, segment_details: Dict,
                              dimension: str) -> str:
        """Generate PhD-level interpretation of mix shift."""
        direction = "increased" if total_change_pct > 0 else "decreased"
        
        biggest_gainer = max(segment_details.items(),
                            key=lambda x: x[1]["change"]["share_change_pct"])
        biggest_loser = min(segment_details.items(),
                           key=lambda x: x[1]["change"]["share_change_pct"])
        
        interpretation = f"📊 MIX SHIFT DECOMPOSITION\n\n"
        interpretation += f"Total change: {total_change_pct:+.1f}%\n\n"
        
        if abs(rate_effect_pct) > abs(mix_effect_pct) * 2:
            interpretation += (
                f"✅ RATE-DRIVEN CHANGE ({abs(rate_effect_pct):.0f}% of change)\n"
                f"This is 'real' growth — {dimension} segments are actually performing "
                f"{'better' if total_change_pct > 0 else 'worse'}.\n\n"
            )
        elif abs(mix_effect_pct) > abs(rate_effect_pct) * 2:
            interpretation += (
                f"⚠️ MIX-DRIVEN CHANGE ({abs(mix_effect_pct):.0f}% of change)\n"
                f"This is composition shift — the {direction} is largely due to "
                f"changing {dimension} mix, NOT real improvement.\n\n"
            )
        else:
            interpretation += (
                f"📊 BALANCED CHANGE\n"
                f"Rate effect: {rate_effect_pct:.0f}% | Mix effect: {mix_effect_pct:.0f}%\n"
                f"Both segment performance and composition are changing.\n\n"
            )
        
        gainer_name = biggest_gainer[0]
        gainer_share = biggest_gainer[1]["change"]["share_change_pct"]
        loser_name = biggest_loser[0]
        loser_share = biggest_loser[1]["change"]["share_change_pct"]
        
        interpretation += (
            f"KEY SHIFTS:\n"
            f"• {gainer_name}: {gainer_share:+.1f}% share change\n"
            f"• {loser_name}: {loser_share:+.1f}% share change\n"
        )
        
        return interpretation
    
    # ═══════════════════════════════════════════════════════════════════════════
    # 🎭 SUBTASK 3.2.3: PARADOX EXPLAINER (LLM-Powered)
    # ═══════════════════════════════════════════════════════════════════════════
    
    def explain_paradox(self, user_id: str, metric_name: str, dimension: str,
                        use_llm: bool = True) -> Dict:
        """
        SUBTASK 3.2.3: Generate comprehensive paradox explanation.
        
        Combines Simpson's Paradox scan + Mix Shift analysis into a
        compelling, actionable narrative.
        """
        try:
            paradox_result = self.scan_simpsons_paradox(user_id, metric_name, dimension)
            mix_shift_result = self.analyze_mix_shift_detailed(user_id, metric_name, dimension)
            
            if "error" in paradox_result:
                return {"error": paradox_result["error"]}
            if "error" in mix_shift_result:
                return {"error": mix_shift_result["error"]}
            
            has_paradox = paradox_result.get("paradox_detected", False)
            
            findings = self._build_structured_findings(paradox_result, mix_shift_result)
            recommendations = self._generate_recommendations(
                has_paradox, paradox_result, mix_shift_result, dimension
            )
            executive_summary = self._generate_executive_summary(
                has_paradox, paradox_result, mix_shift_result,
                metric_name, dimension
            )
            
            if use_llm:
                narrative = self._generate_llm_narrative(
                    findings, recommendations, executive_summary,
                    metric_name, dimension
                )
            else:
                narrative = self._generate_template_narrative(
                    findings, recommendations, executive_summary,
                    metric_name, dimension
                )
            
            result = {
                "metric_name": metric_name,
                "dimension": dimension,
                "has_paradox": has_paradox,
                "paradox_severity": paradox_result.get("severity") if has_paradox else None,
                "executive_summary": executive_summary,
                "detailed_findings": findings,
                "recommendations": recommendations,
                "narrative": narrative,
                "raw_data": {
                    "paradox_scan": paradox_result,
                    "mix_shift_analysis": mix_shift_result
                }
            }
            
            logger.info(f"🎭 Paradox explanation for user {user_id[:8]}...: paradox={has_paradox}")
            return result
            
        except Exception as e:
            logger.error(f"❌ Paradox explainer failed: {e}")
            return {"error": str(e)}
    
    def _build_structured_findings(self, paradox: Dict, mix_shift: Dict) -> Dict:
        """Build structured findings from analysis results."""
        overall = paradox.get("overall_trend", {})
        segments = paradox.get("segment_trends", {})
        decomp = mix_shift.get("decomposition", {})
        
        up_count = sum(1 for s in segments.values() if s.get("growth_pct", 0) > 0)
        down_count = sum(1 for s in segments.values() if s.get("growth_pct", 0) < 0)
        flat_count = len(segments) - up_count - down_count
        
        if segments:
            best_segment = max(segments.items(), key=lambda x: x[1].get("growth_pct", 0))
            worst_segment = min(segments.items(), key=lambda x: x[1].get("growth_pct", 0))
            biggest_share_gain = max(segments.items(), key=lambda x: x[1].get("share_change_pct", 0))
            biggest_share_loss = min(segments.items(), key=lambda x: x[1].get("share_change_pct", 0))
        else:
            best_segment = worst_segment = biggest_share_gain = biggest_share_loss = (None, {})
        
        return {
            "overall_change": {
                "direction": overall.get("direction"),
                "pct": overall.get("growth_pct")
            },
            "segment_summary": {
                "total_segments": len(segments),
                "growing": up_count,
                "declining": down_count,
                "flat": flat_count
            },
            "standout_segments": {
                "best_performer": {
                    "name": best_segment[0],
                    "growth_pct": best_segment[1].get("growth_pct")
                },
                "worst_performer": {
                    "name": worst_segment[0],
                    "growth_pct": worst_segment[1].get("growth_pct")
                },
                "biggest_share_gain": {
                    "name": biggest_share_gain[0],
                    "share_change_pct": biggest_share_gain[1].get("share_change_pct")
                },
                "biggest_share_loss": {
                    "name": biggest_share_loss[0],
                    "share_change_pct": biggest_share_loss[1].get("share_change_pct")
                }
            },
            "change_drivers": {
                "rate_effect_pct": decomp.get("rate_effect", {}).get("pct_of_change"),
                "mix_effect_pct": decomp.get("mix_effect", {}).get("pct_of_change"),
                "primary_driver": "rate" if abs(decomp.get("rate_effect", {}).get("pct_of_change", 0)) >
                                            abs(decomp.get("mix_effect", {}).get("pct_of_change", 0)) else "mix"
            }
        }
    
    def _generate_recommendations(self, has_paradox: bool, paradox: Dict,
                                   mix_shift: Dict, dimension: str) -> List[Dict]:
        """Generate actionable recommendations based on findings."""
        recommendations = []
        
        if has_paradox:
            conflicting = paradox.get("conflicting_segments", [])
            
            recommendations.append({
                "priority": "🔴 HIGH",
                "action": f"Investigate {dimension} segments individually",
                "reason": "Overall trend is misleading due to Simpson's Paradox",
                "specific": f"Focus on: {', '.join(conflicting[:3])}" if conflicting else None
            })
            
            recommendations.append({
                "priority": "🔴 HIGH",
                "action": "Do NOT use overall metric for decision making",
                "reason": "Aggregate number hides segment-level reality",
                "specific": "Report segment-level trends to stakeholders instead"
            })
        
        decomp = mix_shift.get("decomposition", {})
        mix_pct = abs(decomp.get("mix_effect", {}).get("pct_of_change", 0))
        
        if mix_pct > 50:
            recommendations.append({
                "priority": "🟡 MEDIUM",
                "action": "Monitor composition changes closely",
                "reason": f"{mix_pct:.0f}% of change is from mix shift, not real performance",
                "specific": f"Track {dimension} share trends monthly"
            })
        
        top_contributors = mix_shift.get("top_contributors", [])
        if top_contributors:
            top = top_contributors[0]
            if abs(top.get("contribution_pct", 0)) > 60:
                recommendations.append({
                    "priority": "🟡 MEDIUM",
                    "action": f"Reduce dependency on {top['segment']}",
                    "reason": f"Single {dimension} drives {abs(top['contribution_pct']):.0f}% of total change",
                    "specific": "Diversification reduces risk"
                })
        
        if not has_paradox:
            recommendations.append({
                "priority": "🟢 LOW",
                "action": "Continue current monitoring approach",
                "reason": "Overall trend is consistent with segment trends",
                "specific": "No Simpson's Paradox detected"
            })
        
        return recommendations
    
    def _generate_executive_summary(self, has_paradox: bool, paradox: Dict,
                                     mix_shift: Dict, metric_name: str,
                                     dimension: str) -> str:
        """Generate one-paragraph executive summary."""
        overall = paradox.get("overall_trend", {})
        overall_pct = overall.get("growth_pct", 0)
        overall_dir = "increased" if overall_pct > 0 else "decreased"
        
        decomp = mix_shift.get("decomposition", {})
        rate_pct = decomp.get("rate_effect", {}).get("pct_of_change", 0)
        mix_pct = decomp.get("mix_effect", {}).get("pct_of_change", 0)
        
        if has_paradox:
            conflicting = paradox.get("conflicting_segments", [])
            severity = paradox.get("severity", "moderate")
            
            return (
                f"⚠️ ALERT: Simpson's Paradox detected ({severity} severity). "
                f"While {metric_name} appears to have {overall_dir} by {abs(overall_pct):.1f}% overall, "
                f"this is misleading. {len(conflicting)} out of {len(paradox.get('segment_trends', {}))} "
                f"{dimension} segments are moving in the OPPOSITE direction. "
                f"The apparent {'growth' if overall_pct > 0 else 'decline'} is driven by "
                f"composition shift ({abs(mix_pct):.0f}% mix effect), not genuine performance improvement. "
                f"Executive action required: Review {dimension}-level data before making decisions."
            )
        else:
            primary = "segment performance" if abs(rate_pct) > abs(mix_pct) else "composition changes"
            
            return (
                f"✅ {metric_name} {overall_dir} by {abs(overall_pct):.1f}% with no paradox detected. "
                f"Segment trends are consistent with the overall trend. "
                f"The change is primarily driven by {primary} "
                f"({abs(rate_pct):.0f}% rate effect, {abs(mix_pct):.0f}% mix effect). "
                f"This indicates {'genuine performance change' if abs(rate_pct) > abs(mix_pct) else 'shifting composition'} "
                f"across {dimension} segments."
            )
    
    def _generate_llm_narrative(self, findings: Dict, recommendations: List[Dict],
                                 executive_summary: str, metric_name: str,
                                 dimension: str) -> str:
        """
        Generate LLM-powered narrative using NVIDIA Router.

        🚀 GTC 2026: Uses unified NVIDIA LLM Router (Nemotron primary, Groq fallback)
        """
        try:
            from tools.llm_router_client import LLMRouterClient

            prompt = self._build_narrative_prompt(
                findings, recommendations, executive_summary, metric_name, dimension
            )

            # 🚀 GTC 2026: Use NVIDIA Router (Tier 2 = Ultra 253B for PhD-level narrative)
            router = LLMRouterClient(tier=2)
            result = router.call(prompt)

            # Big Tech Pattern: ChatCompletion uses choices[0].message.content
            if result and hasattr(result, 'choices') and result.choices:
                logger.info("✅ NVIDIA Router: Paradox narrative generated")
                return result.choices[0].message.content

            # Fallback to template if LLM fails
            logger.warning("⚠️ NVIDIA Router returned no data, using template")
            return self._generate_template_narrative(
                findings, recommendations, executive_summary, metric_name, dimension
            )

        except Exception as e:
            logger.warning(f"⚠️ LLM narrative failed, using template: {e}")
            return self._generate_template_narrative(
                findings, recommendations, executive_summary, metric_name, dimension
            )
    
    def _build_narrative_prompt(self, findings: Dict, recommendations: List[Dict],
                                 executive_summary: str, metric_name: str,
                                 dimension: str) -> str:
        """Build prompt for LLM narrative generation."""
        return f"""You are a senior business analyst at McKinsey. Write a compelling 3-paragraph narrative 
explaining these data findings to a CEO. Be specific with numbers. Be direct about implications.

METRIC: {metric_name}
DIMENSION ANALYZED: {dimension}

EXECUTIVE SUMMARY:
{executive_summary}

KEY FINDINGS:
- Overall change: {findings['overall_change']['pct']}% ({findings['overall_change']['direction']})
- Segments: {findings['segment_summary']['growing']} growing, {findings['segment_summary']['declining']} declining
- Best performer: {findings['standout_segments']['best_performer']['name']} at {findings['standout_segments']['best_performer']['growth_pct']}%
- Worst performer: {findings['standout_segments']['worst_performer']['name']} at {findings['standout_segments']['worst_performer']['growth_pct']}%
- Primary driver: {findings['change_drivers']['primary_driver']} effect ({findings['change_drivers']['rate_effect_pct']}% rate, {findings['change_drivers']['mix_effect_pct']}% mix)

TOP RECOMMENDATIONS:
{json.dumps([r['action'] for r in recommendations[:3]], indent=2)}

Write the narrative in a professional but accessible tone. Include specific numbers.
Do NOT use bullet points. Write in flowing paragraphs.
"""
    
    def _generate_template_narrative(self, findings: Dict, recommendations: List[Dict],
                                      executive_summary: str, metric_name: str,
                                      dimension: str) -> str:
        """Generate template-based narrative (fallback when no LLM)."""
        overall = findings["overall_change"]
        segments = findings["segment_summary"]
        standouts = findings["standout_segments"]
        drivers = findings["change_drivers"]
        
        narrative = f"""## Analysis of {metric_name} by {dimension}

{executive_summary}

### Performance Breakdown

The analysis reveals significant variation across {dimension} segments. Of the {segments['total_segments']} segments analyzed, {segments['growing']} showed growth while {segments['declining']} experienced decline. The standout performer was {standouts['best_performer']['name']} with {standouts['best_performer']['growth_pct']:+.1f}% growth, while {standouts['worst_performer']['name']} lagged at {standouts['worst_performer']['growth_pct']:+.1f}%.

The change was primarily driven by {'actual segment performance improvements' if drivers['primary_driver'] == 'rate' else 'shifts in segment composition'} (rate effect: {drivers['rate_effect_pct']:.0f}%, mix effect: {drivers['mix_effect_pct']:.0f}%). {'This suggests genuine operational improvement across segments.' if drivers['primary_driver'] == 'rate' else 'This suggests the overall number may be misleading without segment-level context.'}

### Recommended Actions

"""
        for i, rec in enumerate(recommendations[:3], 1):
            narrative += f"{i}. **{rec['priority']}**: {rec['action']} — {rec['reason']}\n"
        
        return narrative


if __name__ == "__main__":
    import sys
    
    print("🎭 Testing ParadoxDetector...")
    
    try:
        detector = ParadoxDetector()
        print("✅ ParadoxDetector initialized")
        
        test_user = sys.argv[1] if len(sys.argv) > 1 else None
        
        if test_user:
            metrics = detector._get_user_metrics(test_user)
            if metrics:
                dims = detector._get_metric_dimensions(test_user, metrics[0])
                if dims:
                    result = detector.scan_simpsons_paradox(test_user, metrics[0], dims[0])
                    print(f"✅ Paradox scan: detected={result.get('paradox_detected', 'N/A')}")
        
        print("\n✅ paradox.py working correctly!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

