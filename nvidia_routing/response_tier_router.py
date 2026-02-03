"""
3-Tier Response Router

Design Pattern: Google Answer Box / ChatGPT Direct Response
- Tier 0: DIRECT answers for simple queries (COUNT, SUM, AVG with 1 row)
- Tier 1: ENHANCED answers for medium complexity (GROUP BY with 2-10 rows)
- Tier 2: FULL PhD analysis for complex queries (correlation, HHI, >10 rows)

Design Pattern: Strategy Pattern
- Separates routing logic from response formatting
- Each tier has its own formatting strategy
- Easy to add new tiers without modifying existing code

Inspired by:
- Google Search (answer box for simple queries, full results for complex)
- ChatGPT (direct answers for math, detailed analysis for essays)
- Netflix (quick recommendations vs. full browse experience)
"""

from typing import Dict, Any, Optional, Literal
import logging

# Type aliases for clarity
ResponseTier = Literal['direct', 'enhanced', 'full_phd']

logger = logging.getLogger(__name__)


class ResponseTierRouter:
    """
    Routes SQL query results to appropriate response tier based on complexity.

    Pattern: Netflix/Spotify - data-driven routing decisions
    """

    # Configuration constants (easy to tune)
    SIMPLE_TEMPLATES = {
        'user_generic_count',
        'user_generic_total',
        'user_generic_avg',
        'user_generic_min',
        'user_generic_max'
    }

    PHD_ANALYSIS_TYPES = {
        'correlation',
        'concentration',
        'comparison',
        'outliers',
        'trend',
        'simpsons_paradox',
        'paradox'
    }

    ENHANCED_ROW_THRESHOLD = 10  # 2-10 rows = enhanced, >10 = full_phd

    @classmethod
    def get_response_tier(
        cls,
        sql_data: Dict[str, Any],
        template_name: str,
        analysis_type: str
    ) -> ResponseTier:
        """
        Determine which response tier to use based on query complexity.

        Args:
            sql_data: SQL query result with 'data', 'row_count', etc.
            template_name: SQL template used (e.g., 'user_generic_count')
            analysis_type: Analysis type from LLM ('basic', 'correlation', etc.)

        Returns:
            'direct' | 'enhanced' | 'full_phd'

        Examples:
            >>> get_response_tier({'row_count': 1, 'data': [...]}, 'user_generic_count', 'basic')
            'direct'

            >>> get_response_tier({'row_count': 5, 'data': [...]}, 'user_generic_avg_by_dimension', 'basic')
            'enhanced'

            >>> get_response_tier({'row_count': 2, 'data': [...]}, 'user_generic_avg_by_dimension', 'correlation')
            'full_phd'
        """
        row_count = sql_data.get('row_count', 0)

        # Guard: No data -> cannot determine tier
        if row_count == 0 or not sql_data.get('data'):
            logger.warning(f"No data in sql_result, defaulting to enhanced tier")
            return 'enhanced'

        # =================================================================
        # Tier 0: DIRECT (Google Answer Box Pattern)
        # =================================================================
        # Single number result from simple aggregation
        # Example: "How many employees?" -> "You have 320 employees"
        # =================================================================
        if cls._is_direct_answer_eligible(template_name, row_count, analysis_type):
            logger.info(f"Tier 0 (DIRECT): template={template_name}, rows={row_count}")
            return 'direct'

        # =================================================================
        # Tier 2: FULL PhD (Statistical Analysis Required)
        # =================================================================
        # Complex statistical analysis needed (correlation, HHI, t-test)
        # OR large dataset requiring aggregation/insights
        # =================================================================
        if cls._requires_phd_analysis(analysis_type, row_count):
            logger.info(f"Tier 2 (FULL PhD): analysis={analysis_type}, rows={row_count}")
            return 'full_phd'

        # =================================================================
        # Tier 1: ENHANCED (Light Analysis + Context)
        # =================================================================
        # Medium complexity - add context but skip heavy PhD analysis
        # Example: "Revenue by department" (5 rows) -> show table + "Dept A is highest"
        # =================================================================
        logger.info(f"Tier 1 (ENHANCED): template={template_name}, rows={row_count}")
        return 'enhanced'

    @classmethod
    def _is_direct_answer_eligible(
        cls,
        template_name: str,
        row_count: int,
        analysis_type: str
    ) -> bool:
        """
        Check if query qualifies for direct answer (Tier 0).

        Criteria:
        - Simple aggregation template (COUNT, SUM, AVG)
        - Single row result
        - Basic analysis type (no statistical tests)
        """
        return (
            template_name in cls.SIMPLE_TEMPLATES
            and row_count == 1
            and analysis_type == 'basic'
        )

    @classmethod
    def _requires_phd_analysis(cls, analysis_type: str, row_count: int) -> bool:
        """
        Check if query requires full PhD-level statistical analysis.

        Criteria:
        - Explicit statistical analysis type (correlation, HHI, etc.)
        - OR large dataset (>10 rows) requiring aggregation
        """
        return (
            analysis_type in cls.PHD_ANALYSIS_TYPES
            or row_count > cls.ENHANCED_ROW_THRESHOLD
        )

    @classmethod
    def format_direct_answer(
        cls,
        sql_data: Dict[str, Any],
        template_name: str,
        user_question: str
    ) -> str:
        """
        Format Tier 0 direct answers in natural language.

        Pattern: Conversational AI (ChatGPT/Claude)
        - Natural language, not technical jargon
        - Contextual (mirrors user's question)
        - Concise (one sentence)

        Args:
            sql_data: SQL query result
            template_name: Template used
            user_question: Original user question

        Returns:
            Natural language answer string

        Examples:
            >>> format_direct_answer({'data': [{'row_count': 320}]}, 'user_generic_count', 'How many employees?')
            'You have 320 employees'

            >>> format_direct_answer({'data': [{'total': 1500000}]}, 'user_generic_total', 'What is total revenue?')
            'Total revenue: 1,500,000.00'
        """
        if not sql_data.get('data'):
            logger.error(f"Cannot format direct answer: no data in sql_result")
            return "No data found"

        result_data = sql_data['data'][0]
        result_value = list(result_data.values())[0] if result_data else 0

        # Handle None/NULL results
        if result_value is None:
            return "No data found for your query"

        # Format based on template type
        if template_name == 'user_generic_count':
            return cls._format_count_answer(result_value, user_question)

        elif template_name == 'user_generic_total':
            return cls._format_total_answer(result_value, sql_data)

        elif template_name == 'user_generic_avg':
            return cls._format_average_answer(result_value, sql_data)

        elif template_name == 'user_generic_min':
            return cls._format_min_answer(result_value, sql_data)

        elif template_name == 'user_generic_max':
            return cls._format_max_answer(result_value, sql_data)

        else:
            # Fallback for unknown template
            logger.warning(f"Unknown template for direct answer: {template_name}")
            return f"Result: {result_value}"

    @staticmethod
    def _format_count_answer(count: int, user_question: str) -> str:
        """
        Format COUNT answers naturally.

        Pattern: Extract noun from question and use in answer
        "How many employees?" -> "You have 320 employees"
        "How many employees do we have?" -> "You have 320 employees"
        "What's the count of records?" -> "You have 1,500 records"
        """
        try:
            # Extract noun from question
            words = user_question.lower().rstrip('?').split()
            noun = "records"  # Default fallback

            # Pattern 1: "how many X" -> extract word after "many"
            if 'many' in words:
                many_idx = words.index('many')
                if many_idx + 1 < len(words):
                    noun = words[many_idx + 1]
            # Pattern 2: "count of X" -> extract word after "of"
            elif 'of' in words:
                of_idx = words.index('of')
                if of_idx + 1 < len(words):
                    noun = words[of_idx + 1]
            # Pattern 3: "total X" -> extract word after "total"
            elif 'total' in words:
                total_idx = words.index('total')
                if total_idx + 1 < len(words):
                    noun = words[total_idx + 1]

            # Handle plural/singular
            count_int = int(count)
            if count_int == 1 and noun.endswith('s'):
                noun = noun[:-1]  # "employees" -> "employee"

            return f"You have {count_int:,} {noun}"
        except (ValueError, IndexError):
            return f"Count: {int(count):,}"

    @staticmethod
    def _format_total_answer(total: float, sql_data: Dict[str, Any]) -> str:
        """Format SUM/TOTAL answers with proper number formatting."""
        metric = sql_data.get('user_requested_metric', 'total')

        # Format large numbers with commas
        if abs(total) >= 1000:
            return f"Total {metric}: {total:,.2f}"
        else:
            return f"Total {metric}: {total:.2f}"

    @staticmethod
    def _format_average_answer(avg: float, sql_data: Dict[str, Any]) -> str:
        """Format AVG/MEAN answers with proper number formatting."""
        metric = sql_data.get('user_requested_metric', 'value')
        return f"Average {metric}: {avg:,.2f}"

    @staticmethod
    def _format_min_answer(min_val: float, sql_data: Dict[str, Any]) -> str:
        """Format MIN answers."""
        metric = sql_data.get('user_requested_metric', 'value')
        return f"Minimum {metric}: {min_val:,.2f}"

    @staticmethod
    def _format_max_answer(max_val: float, sql_data: Dict[str, Any]) -> str:
        """Format MAX answers."""
        metric = sql_data.get('user_requested_metric', 'value')
        return f"Maximum {metric}: {max_val:,.2f}"


# =============================================================================
# Utility Functions (for backward compatibility)
# =============================================================================

def should_skip_phd_analysis(
    sql_data: Dict[str, Any],
    template_name: str,
    analysis_type: str
) -> bool:
    """
    Backward-compatible function for existing code.

    Returns True if query should skip PhD analysis (Tier 0 direct answer)
    """
    tier = ResponseTierRouter.get_response_tier(sql_data, template_name, analysis_type)
    return tier == 'direct'


def format_simple_answer(
    sql_data: Dict[str, Any],
    template_name: str,
    user_question: str
) -> Optional[str]:
    """
    Backward-compatible function for existing code.

    Returns formatted answer if eligible for direct response, else None
    """
    tier = ResponseTierRouter.get_response_tier(
        sql_data,
        template_name,
        sql_data.get('analysis_type', 'basic')
    )

    if tier == 'direct':
        return ResponseTierRouter.format_direct_answer(sql_data, template_name, user_question)

    return None


# =============================================================================
# Configuration (Easy tuning for different deployment environments)
# =============================================================================

class TierConfig:
    """
    Configurable thresholds for different environments.

    Pattern: Feature flags / environment-based config
    - Production: Conservative thresholds (fast direct answers)
    - Staging: Aggressive PhD analysis (test new features)
    - Development: Balanced
    """

    PRODUCTION = {
        'enhanced_row_threshold': 10,
        'enable_tier0': True,
        'enable_tier1': True,
    }

    STAGING = {
        'enhanced_row_threshold': 5,  # More aggressive PhD analysis
        'enable_tier0': True,
        'enable_tier1': True,
    }

    DEVELOPMENT = {
        'enhanced_row_threshold': 10,
        'enable_tier0': True,
        'enable_tier1': True,
    }

    @classmethod
    def load_config(cls, environment: str = 'production') -> Dict[str, Any]:
        """Load configuration for specific environment."""
        configs = {
            'production': cls.PRODUCTION,
            'staging': cls.STAGING,
            'development': cls.DEVELOPMENT,
        }
        return configs.get(environment, cls.PRODUCTION)
