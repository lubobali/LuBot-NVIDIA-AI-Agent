"""
4-Tier Intent Classification System

Tier 0: PhD Analysis Detector (0ms) - Deterministic keyword detection for correlation/concentration/comparison
Tier 1: Regex (0ms) - Exact matches, 80% of queries
Tier 2: Embeddings (5ms) - Semantic similarity, 15% of queries
Tier 3: LLM (100ms) - Ambiguous cases, 5% of queries

Design Pattern: Cheapest method that works -> expensive fallback
"""

import re
from typing import Optional, Tuple
import numpy as np

from intent_routing.correlation_detector import detect_phd_analysis_type

# =============================================================================
# SUPERLATIVE DETECTION (Route "which is best/most/highest" to TEXT)
# =============================================================================
SUPERLATIVE_PATTERNS = [
    r'\b(which|what)\b.*\b(most|highest|best|top|lowest|worst|least|biggest|smallest|largest|greatest)\b',
    r'\b(most|highest|best|top|lowest|worst|least|biggest|smallest|largest|greatest)\b.*\b(which|what)\b',
    r'\btop\s*(1|one)\b',
    r'\b(the|is)\s+(most|highest|best|lowest|worst|least|biggest|smallest|largest|greatest)\b',
    r'\b(number\s*1|#1|number\s*one)\b',
]

SUPERLATIVE_EXCLUSIONS = ['show', 'chart', 'graph', 'visualize', 'plot', 'heatmap', 'histogram', 'bar chart', 'pie chart', 'line chart', 'display', 'draw']

def is_superlative_question(query: str) -> bool:
    """
    Detect questions asking for #1 result (text answer, not chart).

    Examples:
        "which month is most profit" -> True
        "what product has highest sales" -> True
        "show me profit by month" -> False
    """
    query_lower = query.lower()

    for exclusion in SUPERLATIVE_EXCLUSIONS:
        if exclusion in query_lower:
            return False

    for pattern in SUPERLATIVE_PATTERNS:
        if re.search(pattern, query_lower):
            return True

    return False

class IntentClassifier:
    """
    Production-grade intent classification.
    Pattern: Cheapest method that works -> expensive fallback
    """

    # =========================================================================
    # INTENT TYPES (Single Source of Truth)
    # =========================================================================
    GREETING = "GREETING"
    IDENTITY = "IDENTITY"
    MEMORY_RECALL = "MEMORY_RECALL"
    MEMORY_UPDATE = "MEMORY_UPDATE"
    DATA_MODE_SWITCH = "DATA_MODE_SWITCH"
    DATA_QUERY = "DATA_QUERY"
    WEB_SEARCH = "WEB_SEARCH"
    DOCUMENT_QA = "DOCUMENT_QA"
    HELP_REQUEST = "HELP_REQUEST"
    GENERAL = "GENERAL"
    CAPABILITIES = "CAPABILITIES"

    # =========================================================================
    # FOLLOWUP INTENTS (Require Previous Context)
    # =========================================================================
    ADVICE_REQUEST = "ADVICE_REQUEST"
    FOLLOWUP = "FOLLOWUP"
    CLARIFICATION = "CLARIFICATION"
    DEEP_DIVE = "DEEP_DIVE"
    COMPARISON = "COMPARISON"

    # =========================================================================
    # PREDICTION INTENT (Forecast/Prophet patterns)
    # =========================================================================
    PREDICTION = "PREDICTION"

    # =========================================================================
    # DOCUMENT_GENERATION INTENT (Export/Report patterns)
    # =========================================================================
    DOCUMENT_GENERATION = "DOCUMENT_GENERATION"

    # =========================================================================
    # DATA_LIBRARY INTENT (Table preview/schema requests)
    # =========================================================================
    DATA_LIBRARY = "DATA_LIBRARY"

    # =========================================================================
    # TIER 1: REGEX PATTERNS (0ms, $0)
    # =========================================================================
    TIER1_PATTERNS = [
        # Greetings (exact or start-of-message only)
        (r"^(hi|hello|hey|yo)$", GREETING, True),
        (r"^(hi|hello|hey|yo)[!\s,.]", GREETING, False),
        (r"^good (morning|afternoon|evening)", GREETING, False),
        (r"^what'?s up\??$", GREETING, True),

        # Personal questions ("my X" = memory lookup, NOT identity)
        (r"who('?s| is) my (boss|manager|supervisor|lead|director|mentor)", GENERAL, False),
        (r"who do i (report to|work for|work with)", GENERAL, False),
        (r"who('?s| is) my", GENERAL, False),
        (r"what('?s| is) my (project|task|goal|deadline|job|role)", GENERAL, False),
        (r"where do i work", GENERAL, False),
        (r"what company do i", GENERAL, False),
        (r"my (boss|manager|supervisor|employer|company|project)", GENERAL, False),
        (r"tell me about my (boss|manager|project|work|job)", GENERAL, False),

        # Identity questions (must come AFTER personal questions)
        (r"^who (are you|is lubot|is lubo|made you|built you|created you)", IDENTITY, False),
        (r"^what (are you|is lubot)", IDENTITY, False),
        (r"^tell me about yourself", IDENTITY, False),

        # CAPABILITIES
        (r"^help$", CAPABILITIES, True),
        (r"^what can you do", CAPABILITIES, False),
        (r"^what are your (capabilities|features|skills|functions)", CAPABILITIES, False),
        (r"^how can you (help|assist)", CAPABILITIES, False),
        (r"^show me (your )?(features|capabilities)", CAPABILITIES, False),
        (r"^what do you (do|offer|provide)", CAPABILITIES, False),
        (r"how smart are you", CAPABILITIES, False),

        # Memory recall (explicit)
        (r"what do you (know|remember) about me", MEMORY_RECALL, False),
        (r"what have i told you", MEMORY_RECALL, False),
        (r"do you remember", MEMORY_RECALL, False),
        (r"my memories", MEMORY_RECALL, False),

        # Memory update (explicit)
        (r"^(remember that|don'?t forget|forget that|update.*(memory|remember))", MEMORY_UPDATE, False),

        # Data mode switch
        (r"(switch to|use) (demo|my data)", DATA_MODE_SWITCH, False),
        (r"(show|use) (demo|uploaded|my) data", DATA_MODE_SWITCH, False),

        # PREDICTION PATTERNS
        (r"predict\s+(my\s+)?(traffic|profit|sales|revenue|clicks|data)", PREDICTION, False),
        (r"forecast\s+(my\s+)?(traffic|profit|sales|revenue|clicks|data)", PREDICTION, False),
        (r"what will\s+(my\s+)?(traffic|profit|sales|revenue|clicks)", PREDICTION, False),
        (r"(predict|forecast).*(next|coming)\s+(week|month|quarter|year)", PREDICTION, False),
        (r"(next|coming)\s+(week|month|quarter|year).*(predict|forecast)", PREDICTION, False),

        # DOCUMENT_GENERATION PATTERNS
        (r"generate\s+(a\s+)?(pdf|docx|excel|word)\s*(report|document|file)?", DOCUMENT_GENERATION, False),
        (r"create\s+(a\s+)?(pdf|docx|excel|word)\s*(report|document|file)?", DOCUMENT_GENERATION, False),
        (r"make\s+(me\s+)?(a\s+)?(pdf|docx|excel|word)\s*(report|document)?", DOCUMENT_GENERATION, False),
        (r"(pdf|docx|excel|word)\s+report", DOCUMENT_GENERATION, False),
        (r"export\s+(to|as)\s+(pdf|docx|excel|word)", DOCUMENT_GENERATION, False),
        (r"(export|download)\s+(my\s+)?data\s+(to|as)\s+(excel|csv|pdf)", DOCUMENT_GENERATION, False),

        # DATA_LIBRARY PATTERNS
        (r"show\s+(me\s+)?my\s+files?\b", "DATA_LIBRARY", False),
        (r"show\s+(me\s+)?my\s+tables?\b", "DATA_LIBRARY", False),
        (r"what\s+files?\s+do\s+i\s+have", "DATA_LIBRARY", False),
        (r"what\s+tables?\s+do\s+i\s+have", "DATA_LIBRARY", False),
        (r"list\s+(my\s+)?files?", "DATA_LIBRARY", False),
        (r"list\s+(my\s+)?tables?", "DATA_LIBRARY", False),
        (r"show\s+(me\s+)?(the\s+)?(table\s+)?preview", "DATA_LIBRARY", False),
        (r"show\s+(me\s+)?(the\s+)?columns", "DATA_LIBRARY", False),
        (r"what\s+columns\s+(do\s+i\s+have|are\s+there|are\s+in)", "DATA_LIBRARY", False),
        (r"(show|display|list)\s+(me\s+)?(my\s+)?(uploaded\s+)?(files?|data|tables?)", "DATA_LIBRARY", False),
        (r"what('?s|\s+is)\s+in\s+my\s+(data|files?|tables?)", "DATA_LIBRARY", False),
        (r"show\s+(me\s+)?(the\s+)?last\s+(table|file|upload)", "DATA_LIBRARY", False),
        (r"(what|which)\s+(files?|tables?)\s+(do\s+i\s+have|did\s+i\s+upload)", "DATA_LIBRARY", False),
        (r"my\s+(uploaded\s+)?(files?|tables?)", "DATA_LIBRARY", False),
        (r"show\s+(me\s+)?(the\s+)?\w+\s+(table|file)\b", "DATA_LIBRARY", False),
        (r"(table|file)\s+(preview|schema|structure|columns)", "DATA_LIBRARY", False),

        # FOLLOWUP PATTERNS
        (r"what (were|was|are|is) the (recommendations|results|findings|analysis|anomalies|predictions|insights|metrics|numbers)", FOLLOWUP, False),
        (r"(show|list|tell|give) me the (recommendations|results|findings|analysis)", FOLLOWUP, False),
        (r"(summarize|recap|repeat) the (report|recommendations|results|findings)", FOLLOWUP, False),

        # ADVICE_REQUEST patterns
        (r"(suggest|improve|feedback|opinion|recommend|better|enhance)", ADVICE_REQUEST, False),
        (r"what (should|could|can) (i|we)", ADVICE_REQUEST, False),
        (r"how (can|could|should) (i|we) (improve|make|fix)", ADVICE_REQUEST, False),
        (r"any (tips|advice|suggestions)", ADVICE_REQUEST, False),
        (r"give me (feedback|suggestions|advice)", ADVICE_REQUEST, False),

        # FOLLOWUP patterns - Pronouns referencing previous output
        (r"^(this|that|it)\b", FOLLOWUP, False),
        (r"(about|with|for) (this|that|it)\b", FOLLOWUP, False),
        (r"(the|this|that) (report|document|file|chart|result|pdf|docx|excel)", FOLLOWUP, False),
        (r"^what about", FOLLOWUP, False),
        (r"(continue|more on) (this|that)", FOLLOWUP, False),

        # CLARIFICATION patterns
        (r"^why\b", CLARIFICATION, False),
        (r"^how come", CLARIFICATION, False),
        (r"^explain\b", CLARIFICATION, False),
        (r"what (made|caused) you", CLARIFICATION, False),
        (r"reasoning behind", CLARIFICATION, False),
        (r"why did you (say|recommend|suggest|choose)", CLARIFICATION, False),

        # DEEP_DIVE patterns
        (r"tell me more", DEEP_DIVE, False),
        (r"expand on", DEEP_DIVE, False),
        (r"more (details|info|information|about)", DEEP_DIVE, False),
        (r"go deeper", DEEP_DIVE, False),
        (r"elaborate on", DEEP_DIVE, False),
        (r"can you explain .* in (more )?detail", DEEP_DIVE, False),

        # COMPARISON patterns
        (r"what if", COMPARISON, False),
        (r"what would happen", COMPARISON, False),
        (r"\bcompare\b", COMPARISON, False),
        (r"\b(versus|vs\.?)\b", COMPARISON, False),
        (r"instead of", COMPARISON, False),
        (r"difference between", COMPARISON, False),
    ]

    # =========================================================================
    # TIER 2: CANONICAL EMBEDDINGS (5ms, $0.0001)
    # =========================================================================
    TIER2_CANONICALS = {
        GREETING: [
            "hi there",
            "hello",
            "hey how are you",
            "good morning",
            "greetings",
        ],
        IDENTITY: [
            "who are you",
            "what is lubot",
            "tell me about yourself",
            "who created this",
            "who built you",
        ],
        MEMORY_RECALL: [
            "what do you know about me",
            "what do you remember",
            "tell me what you know about me",
            "what facts do you have about me",
            "what have i shared with you",
        ],
        HELP_REQUEST: [
            "help me with",
            "can you help me",
            "i need help with",
            "assist me with",
            "help me prepare",
            "help me create",
        ],
        PREDICTION: [
            "predict my traffic for next week",
            "forecast profit for next month",
            "what will my sales be next quarter",
            "predict revenue for next year",
            "forecast clicks for coming month",
            "what will traffic look like next week",
            "predict my data trends",
        ],
        DOCUMENT_GENERATION: [
            "generate a pdf report",
            "create a word document",
            "make me a report",
            "export to excel",
            "create pdf of my data",
            "generate report of my sales",
            "make a summary document",
            "download my data as excel",
            "export data to spreadsheet",
        ],
        DATA_QUERY: [
            "how many clicks",
            "show me traffic",
            "top pages",
            "clicks yesterday",
            "chart of visits",
            "analytics report",
            "show me a graph",
        ],
        GENERAL: [
            "who is my boss",
            "who is my manager",
            "who is my supervisor",
            "who do I report to",
            "who manages me",
            "tell me about my boss",
            "where do I work",
            "what company do I work for",
            "what company am I at",
            "who is my employer",
            "tell me about my job",
            "what am I working on",
            "what is my current project",
            "what are my tasks",
            "what is my deadline",
            "tell me about my project",
            "who is on my team",
            "who are my colleagues",
            "who do I work with",
            "what do you know about my work",
            "tell me about my situation",
        ],
        WEB_SEARCH: [
            "what is the latest news",
            "search for",
            "look up",
            "find information about",
            "current events",
        ],
        CAPABILITIES: [
            "what can you do",
            "what are your capabilities",
            "what are you capable of",
            "what do you know how to do",
            "how can you help me",
            "how can you assist me",
            "what help can you provide",
            "what features do you have",
            "show me your features",
            "what functions do you have",
            "what are you good at",
            "what are your skills",
            "what can you help with",
            "how smart are you",
            "what do you specialize in",
        ],
        ADVICE_REQUEST: [
            "what suggestions do you have",
            "how can I improve this",
            "give me feedback on the report",
            "what would you recommend",
            "make it better",
            "any tips for improvement",
            "how should I proceed",
            "what changes would you suggest",
        ],
        FOLLOWUP: [
            "tell me more about this",
            "what about the report",
            "continue with that",
            "the document you created",
            "regarding the previous output",
            "about what you just said",
            "can we continue",
            "back to the report",
        ],
        CLARIFICATION: [
            "why did you recommend that",
            "explain the home page suggestion",
            "how come contact page is declining",
            "what made you prioritize that",
            "why is that important",
            "explain your reasoning",
            "what's the logic behind this",
            "why did you choose that approach",
        ],
        DEEP_DIVE: [
            "tell me more about the anomalies",
            "expand on the growth prediction",
            "give me details about traffic trends",
            "go deeper on recommendations",
            "I want more information about this",
            "elaborate on the analysis",
            "can you break this down further",
            "explain in more detail",
        ],
        COMPARISON: [
            "what if we ignore the anomalies",
            "compare this month to last month",
            "what would happen if we focus only on home page",
            "versus focusing on declining pages",
            "how does this compare to before",
            "what's the difference between these options",
            "contrast the two approaches",
            "alternatively what about",
        ],
    }

    # Similarity threshold for Tier 2
    TIER2_THRESHOLD = 0.82

    # =========================================================================
    # TIER 3: LLM PROMPT (100ms, $0.01)
    # =========================================================================
    TIER3_PROMPT = """Classify this user message into ONE category.

MESSAGE: {question}

CATEGORIES:
- GREETING: Simple greetings (hi, hello, hey, good morning)
- IDENTITY: Questions about who/what you are, who built you
- CAPABILITIES: What the AI can do, features, abilities ("what can you do", "help", "your skills")
- MEMORY_RECALL: Questions about what you know/remember about the user
- MEMORY_UPDATE: User wants you to remember/forget something
- DATA_MODE_SWITCH: User wants to switch between demo and their data
- DATA_QUERY: Analytics questions (clicks, traffic, charts, reports, data)
- WEB_SEARCH: Current events, external information, news
- DOCUMENT_QA: Questions about uploaded documents
- HELP_REQUEST: User wants help with a specific task (not data-related)
- GENERAL: Casual conversation, unclear intent

FOLLOWUP CATEGORIES (require previous context):
- ADVICE_REQUEST: User wants suggestions/improvements ("suggest", "improve", "feedback", "recommend")
- FOLLOWUP: References previous output with pronouns ("this", "that", "it", "the report")
- CLARIFICATION: Asks why/how ("why did you", "explain", "how come", "reasoning")
- DEEP_DIVE: Wants more details ("tell me more", "expand on", "details", "elaborate")
- COMPARISON: Hypothetical or comparative ("what if", "compare", "versus", "instead of")

CRITICAL RULES:
- "help me [task]" with data/analytics context -> DATA_QUERY
- "help me [task]" without data context -> HELP_REQUEST
- Questions with "clicks", "traffic", "chart", "report" -> DATA_QUERY
- Simple "help" alone -> CAPABILITIES
- "what can you do", "how can you help", "your features" -> CAPABILITIES

PERSONAL vs IDENTITY (IMPORTANT):
- "who is MY boss/manager/supervisor" -> GENERAL (personal question about USER)
- "what is MY project/company/deadline" -> GENERAL (personal question about USER)
- "who manages ME" or "who hired ME" -> GENERAL (personal question about USER)
- "who are YOU" or "what are YOU" -> IDENTITY (question about the AI)
- "who built YOU" or "who created YOU" -> IDENTITY (question about the AI)
- Rule: "my X" or "X me" = GENERAL, "you/yourself" = IDENTITY

Respond with ONE WORD only:"""

    def __init__(self, embedding_model=None, llm_generator=None):
        """
        Initialize classifier with optional embedding model and LLM.

        Args:
            embedding_model: Model for Tier 2 embeddings (e.g., NVIDIAEmbeddings)
            llm_generator: AdalFlow generator for Tier 3 fallback
        """
        self.embedding_model = embedding_model
        self.llm_generator = llm_generator
        self.canonical_embeddings = {}

        # Pre-compute canonical embeddings if model available
        if self.embedding_model:
            self._precompute_embeddings()

    def _precompute_embeddings(self):
        """Pre-compute embeddings for Tier 2 canonicals."""
        print("Pre-computing canonical embeddings...")
        for intent, examples in self.TIER2_CANONICALS.items():
            embeddings = []
            for example in examples:
                emb = self.embedding_model.encode(example)
                embeddings.append(emb)
            self.canonical_embeddings[intent] = embeddings
        print(f"Pre-computed embeddings for {len(self.canonical_embeddings)} intents")

    def classify(self, question: str) -> Tuple[str, int, float]:
        """
        Classify user intent using 4-tier system.

        Args:
            question: User's message

        Returns:
            Tuple[str, int, float]: (intent, tier_used, confidence)
        """
        question_clean = question.strip()
        question_lower = question_clean.lower()

        # =====================================================================
        # TIER 0: PhD Analysis Detector (0ms, 100% reliable)
        # =====================================================================
        phd_analysis_type = detect_phd_analysis_type(question_clean)
        if phd_analysis_type:
            print(f"Tier 0 (PhD detector): DATA_QUERY (analysis_type={phd_analysis_type}) (1.00)")
            return self.DATA_QUERY, 0, 1.0

        # =====================================================================
        # TIER 1: Regex (0ms)
        # =====================================================================
        intent, confidence = self._tier1_regex(question_lower)
        if intent:
            print(f"Tier 1 (regex): {intent} ({confidence:.2f})")
            return intent, 1, confidence

        # =====================================================================
        # TIER 2: Embeddings (5ms)
        # =====================================================================
        if self.embedding_model and self.canonical_embeddings:
            intent, confidence = self._tier2_embeddings(question_lower)
            if intent and confidence >= self.TIER2_THRESHOLD:
                print(f"Tier 2 (embedding): {intent} ({confidence:.2f})")
                return intent, 2, confidence

        # =====================================================================
        # TIER 3: LLM Fallback (100ms)
        # =====================================================================
        if self.llm_generator:
            intent, confidence = self._tier3_llm(question_clean)
            print(f"Tier 3 (LLM): {intent} ({confidence:.2f})")
            return intent, 3, confidence

        # =====================================================================
        # FALLBACK: Default to GENERAL
        # =====================================================================
        print(f"No classification, defaulting to GENERAL")
        return self.GENERAL, 0, 0.5

    def _tier1_regex(self, question_lower: str) -> Tuple[Optional[str], float]:
        """Tier 1: Fast regex matching."""
        for pattern, intent, must_be_exact in self.TIER1_PATTERNS:
            if must_be_exact:
                if re.match(pattern, question_lower):
                    return intent, 1.0
            else:
                if re.search(pattern, question_lower):
                    return intent, 0.95
        return None, 0.0

    def _tier2_embeddings(self, question_lower: str) -> Tuple[Optional[str], float]:
        """Tier 2: Semantic similarity matching."""
        try:
            query_embedding = self.embedding_model.encode(question_lower)

            best_intent = None
            best_score = 0.0

            for intent, embeddings in self.canonical_embeddings.items():
                for canonical_emb in embeddings:
                    # Cosine similarity
                    score = np.dot(query_embedding, canonical_emb) / (
                        np.linalg.norm(query_embedding) * np.linalg.norm(canonical_emb)
                    )
                    if score > best_score:
                        best_score = score
                        best_intent = intent

            return best_intent, best_score

        except Exception as e:
            print(f"Tier 2 embedding failed: {e}")
            return None, 0.0

    def _tier3_llm(self, question: str) -> Tuple[str, float]:
        """Tier 3: LLM classification fallback."""
        try:
            prompt = self.TIER3_PROMPT.format(question=question)
            result = self.llm_generator.call(prompt_kwargs={"input_str": prompt})

            if result:
                intent = result.data.strip().upper() if hasattr(result, 'data') else str(result).strip().upper()

                # Validate intent (including followup intents)
                valid_intents = [
                    self.GREETING, self.IDENTITY, self.CAPABILITIES, self.MEMORY_RECALL,
                    self.MEMORY_UPDATE, self.DATA_MODE_SWITCH, self.DATA_QUERY,
                    self.WEB_SEARCH, self.DOCUMENT_QA, self.HELP_REQUEST, self.GENERAL,
                    self.ADVICE_REQUEST, self.FOLLOWUP, self.CLARIFICATION,
                    self.DEEP_DIVE, self.COMPARISON
                ]

                if intent in valid_intents:
                    return intent, 0.85

            return self.GENERAL, 0.5

        except Exception as e:
            print(f"Tier 3 LLM failed: {e}")
            return self.GENERAL, 0.5

    def is_followup_intent(self, intent: str) -> bool:
        """
        Check if intent requires previous context.

        Args:
            intent: Classified intent type

        Returns:
            bool: True if this intent needs context from previous outputs
        """
        followup_intents = {
            self.ADVICE_REQUEST,
            self.FOLLOWUP,
            self.CLARIFICATION,
            self.DEEP_DIVE,
            self.COMPARISON
        }
        return intent in followup_intents


# =============================================================================
# SINGLETON INSTANCE (Reuse pre-computed embeddings)
# =============================================================================
_classifier_instance = None

def get_intent_classifier(embedding_model=None, llm_generator=None) -> IntentClassifier:
    """Get or create singleton intent classifier."""
    global _classifier_instance

    if _classifier_instance is None:
        _classifier_instance = IntentClassifier(
            embedding_model=embedding_model,
            llm_generator=llm_generator
        )

    return _classifier_instance
