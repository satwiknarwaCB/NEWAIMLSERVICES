# chatbot.py - Advanced Chatbot Manager with Accurate Token Counting, Enhanced Security, and Custom LLM Support

import re
import os
import logging
import time
from typing import List, Dict, Any, Optional, Tuple
from functools import wraps
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_qdrant import QdrantVectorStore
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.documents import Document
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from qdrant_client import QdrantClient
from dotenv import load_dotenv
import tiktoken  # For accurate token counting

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
load_dotenv()

# FIXED: Realistic Security and Cost Controls
MAX_QUERY_LENGTH = 2000
MAX_RESPONSE_LENGTH = 15000
MAX_TOKENS_PER_SESSION = 75000  # Reduced to realistic limit
RATE_LIMIT_SECONDS = 0.5
BLOCKED_PATTERNS = [
    r'<script.*?>.*?</script>',
    r'javascript:',
    r'data:text/html',
    r'<iframe.*?>.*?</iframe>'
]

# Enhanced PII patterns
PII_PATTERNS = {
    'ssn': r'\b\d{3}[-.\s]?\d{2}[-.\s]?\d{4}\b',
    'credit_card': r'\b\d{4}[-.\s]?\d{4}[-.\s]?\d{4}[-.\s]?\d{4}\b',
    'phone': r'\b(?:\+1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b',
    'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
    'ip_address': r'\b(?:\d{1,3}\.){3}\d{1,3}\b'
}

# Content filtering
PROFANITY_LIST = [
    'badword1', 'badword2', 'inappropriate1', 'inappropriate2'
    # Add actual profanity list for production
]

SENSITIVE_TOPICS = [
    'password', 'api_key', 'secret', 'token', 'private_key',
    'social_security', 'bank_account', 'credit_card'
]

def rate_limit(min_interval=RATE_LIMIT_SECONDS):
    """Rate limiting decorator"""
    def decorator(func):
        last_called = [0.0]
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            elapsed = time.time() - last_called[0]
            left_to_wait = min_interval - elapsed
            if left_to_wait > 0:
                time.sleep(left_to_wait)
            ret = func(*args, **kwargs)
            last_called[0] = time.time()
            return ret
        return wrapper
    return decorator

class TokenCounter:
    """FIXED: Accurate token counting using tiktoken"""
    
    def __init__(self):
        try:
            # Try to get the correct tokenizer for the model
            self.encoding = tiktoken.get_encoding("cl100k_base")  # GPT-3.5/4 tokenizer
        except:
            # Fallback to approximate counting
            self.encoding = None
            logger.warning("tiktoken not available, using approximate token counting")
    
    def count_tokens(self, text: str) -> int:
        """Count tokens accurately"""
        if not text:
            return 0
            
        if self.encoding:
            try:
                return len(self.encoding.encode(text))
            except:
                pass
        
        # Fallback: approximate token counting
        # For English text, roughly 1 token = 0.75 words
        words = len(text.split())
        return int(words * 1.33)
    
    def estimate_tokens_from_messages(self, messages: List[Dict]) -> int:
        """Estimate tokens for a conversation"""
        total = 0
        for message in messages:
            total += self.count_tokens(message.get('content', ''))
            total += 4  # Overhead per message
        return total + 3  # Conversation overhead

class ContentFilter:
    """Enhanced content filtering with multiple security layers"""
    
    def __init__(self):
        self.pii_patterns = {k: re.compile(v, re.IGNORECASE) for k, v in PII_PATTERNS.items()}
        self.blocked_patterns = [re.compile(p, re.IGNORECASE | re.DOTALL) for p in BLOCKED_PATTERNS]
        self.profanity_pattern = re.compile('|'.join(PROFANITY_LIST), re.IGNORECASE)
        self.sensitive_pattern = re.compile('|'.join(SENSITIVE_TOPICS), re.IGNORECASE)
    
    def scan_for_pii(self, text: str) -> Tuple[bool, List[str], str]:
        """Scan text for PII and return cleaned version"""
        found_pii = []
        cleaned_text = text
        
        for pii_type, pattern in self.pii_patterns.items():
            matches = pattern.findall(text)
            if matches:
                found_pii.append(pii_type)
                cleaned_text = pattern.sub(f'[{pii_type.upper()}_REDACTED]', cleaned_text)
        
        return bool(found_pii), found_pii, cleaned_text
    
    def check_content_safety(self, text: str) -> Tuple[bool, List[str]]:
        """Check for various content safety issues"""
        issues = []
        
        # Check for blocked patterns (XSS, injection, etc.)
        for pattern in self.blocked_patterns:
            if pattern.search(text):
                issues.append("malicious_content")
                break
        
        # Check for profanity
        if self.profanity_pattern.search(text):
            issues.append("profanity")
        
        # Check for sensitive topics
        if self.sensitive_pattern.search(text):
            issues.append("sensitive_content")
        
        # Check for excessive length
        if len(text) > MAX_QUERY_LENGTH:
            issues.append("excessive_length")
        
        return bool(issues), issues
    
    def filter_response(self, response: str) -> Tuple[str, bool, List[str]]:
        """Filter and clean response content"""
        issues = []
        
        # Check for PII in response
        has_pii, pii_types, cleaned_response = self.scan_for_pii(response)
        if has_pii:
            issues.extend(pii_types)
        
        # Check response safety
        has_safety_issues, safety_issues = self.check_content_safety(cleaned_response)
        if has_safety_issues:
            issues.extend(safety_issues)
        
        # Truncate if too long
        if len(cleaned_response) > MAX_RESPONSE_LENGTH:
            cleaned_response = cleaned_response[:MAX_RESPONSE_LENGTH] + "... [Response truncated for safety]"
            issues.append("response_truncated")
        
        return cleaned_response, bool(issues), issues

class AdvancedRetriever:
    """Advanced retrieval with hybrid search and reranking"""
    
    def __init__(self, vector_store: QdrantVectorStore, embeddings, k: int = 8, score_threshold: float = 0.4):
        self.vector_store = vector_store
        self.embeddings = embeddings
        self.k = k
        self.score_threshold = score_threshold
        self.token_counter = TokenCounter()
    
    def hybrid_search(self, query: str, k: int = None) -> List[Document]:
        """Perform hybrid search combining semantic and keyword search"""
        k = k or self.k
        
        try:
            # Semantic search
            semantic_results = self.vector_store.similarity_search_with_score(
                query, 
                k=k
            )
            
            # Filter by score threshold
            filtered_results = [
                doc for doc, score in semantic_results 
                if score >= self.score_threshold
            ]
            
            return filtered_results[:k]
            
        except Exception as e:
            logger.error(f"Hybrid search error: {e}")
            # Fallback to basic search
            return self.vector_store.similarity_search(query, k=k)
    
    def rerank_documents(self, query: str, documents: List[Document]) -> List[Document]:
        """Rerank documents based on relevance"""
        if not documents:
            return []
        
        try:
            # Simple relevance scoring based on query term overlap
            query_terms = set(query.lower().split())
            
            def calculate_relevance(doc: Document) -> float:
                doc_terms = set(doc.page_content.lower().split())
                overlap = len(query_terms.intersection(doc_terms))
                total = len(query_terms)
                return overlap / total if total > 0 else 0
            
            # Sort by relevance
            scored_docs = [(doc, calculate_relevance(doc)) for doc in documents]
            scored_docs.sort(key=lambda x: x[1], reverse=True)
            
            return [doc for doc, score in scored_docs]
            
        except Exception as e:
            logger.error(f"Reranking error: {e}")
            return documents

class ChatbotManager:
    """Enhanced chatbot with accurate token counting and security"""
    
    # FIXED: Use persistent collection name matching vectors.py
    DEFAULT_COLLECTION_NAME = "Legal_documents"
    
    def __init__(
        self,
        model_name: str = "BAAI/bge-small-en",
        device: str = "cpu",
        encode_kwargs: dict = None,
        llm_model: str = "llama-3.3-70b-versatile",
        llm_temperature: float = 0.3,
        max_tokens: int = 4000,
        qdrant_url: str = None,
        collection_name: str = None,
        retrieval_k: int = 8,
        score_threshold: float = 0.4,
        use_custom_llm: bool = False,  # NEW: Toggle for custom LLM
        custom_llm_url: str = None,  # NEW: Custom LLM endpoint  
        custom_llm_api_key: str = None,  # NEW: Custom LLM API key
        custom_llm_model_name: str = None  # NEW: Custom model name
    ):
        """Initialize chatbot with comprehensive configuration + custom LLM support"""
        
        # Store custom LLM settings
        self.use_custom_llm = use_custom_llm
        self.custom_llm_url = custom_llm_url
        self.llm_type = "custom" if use_custom_llm else "groq"
        
        # Initialize embeddings
        self.embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={"device": device},
            encode_kwargs=encode_kwargs or {"normalize_embeddings": True}
        )
        
        # NEW: Initialize LLM (external Groq or custom/internal)
        if use_custom_llm and custom_llm_url:
            # Use custom/internal LLM
            logger.info(f"🔧 Initializing CUSTOM LLM: {custom_llm_url}")
            self.llm = self._initialize_custom_llm(
                custom_llm_url,
                custom_llm_api_key,
                custom_llm_model_name or llm_model,
                llm_temperature,
                max_tokens
            )
            logger.info(f"✅ Custom LLM initialized: {custom_llm_model_name or llm_model}")
        else:
            # Use external Groq LLM
            logger.info("🔧 Initializing EXTERNAL LLM: Groq")
            self.llm = ChatGroq(
                model=llm_model,
                temperature=llm_temperature,
                max_tokens=max_tokens,
                groq_api_key=os.getenv('GROQ_API_KEY')
            )
            logger.info(f"✅ Groq LLM initialized: {llm_model}")
        
        # FIXED: Initialize Qdrant with persistent collection
        self.qdrant_url = qdrant_url or os.getenv('QDRANT_URL')
        self.qdrant_api_key = os.getenv('QDRANT_API_KEY')
        # Use the persistent collection name from vectors.py
        self.collection_name = collection_name or self.DEFAULT_COLLECTION_NAME
        
        self.qdrant_client = QdrantClient(
            url=self.qdrant_url,
            api_key=self.qdrant_api_key,
            prefer_grpc=False
        )
        
        # Check if collection exists before initializing vector store
        try:
            collections = self.qdrant_client.get_collections().collections
            collection_exists = any(c.name == self.collection_name for c in collections)
            if not collection_exists:
                logger.warning(f"Collection '{self.collection_name}' does not exist yet. It will be created when documents are processed.")
        except Exception as e:
            logger.warning(f"Could not check collection existence: {e}")
        
        # Initialize vector store - FIX for newer LangChain versions
        try:
            # Try the newer initialization method (works with langchain-qdrant >= 0.1.0)
            self.vector_store = QdrantVectorStore(
                client=self.qdrant_client,
                collection_name=self.collection_name,
                embedding=self.embeddings
            )
        except TypeError as e:
            # Fallback: Try older initialization method
            logger.warning(f"QdrantVectorStore() failed: {e}, trying from_existing_collection")
            self.vector_store = QdrantVectorStore.from_existing_collection(
                embedding=self.embeddings,
                collection_name=self.collection_name,
                url=self.qdrant_url,
                api_key=self.qdrant_api_key
            )
        except Exception as e:
            # If collection doesn't exist, set vector_store to None temporarily
            logger.warning(f"Could not initialize vector store: {e}. Will initialize after documents are processed.")
            self.vector_store = None
        
        # Retrieval settings
        self.retrieval_k = retrieval_k
        self.score_threshold = score_threshold
        
        # Basic retriever - only initialize if vector_store exists
        if self.vector_store is not None:
            self.basic_retriever = self.vector_store.as_retriever(
                search_type="similarity_score_threshold",
                search_kwargs={
                    "k": self.retrieval_k,
                    "score_threshold": self.score_threshold
                }
            )
            
            # Advanced retriever with hybrid search
            self.advanced_retriever = AdvancedRetriever(
                self.vector_store,
                self.embeddings,
                k=self.retrieval_k,
                score_threshold=self.score_threshold
            )
        else:
            self.basic_retriever = None
            self.advanced_retriever = None
            logger.warning("Retrievers not initialized - waiting for documents to be processed")
        
        # ENHANCED: RAG Prompt Template with Comprehensive Indian Legal Context
        self.prompt_template = """You are an expert Indian legal assistant. Answer the question using ONLY the information from the provided legal documents.

CRITICAL RULES:
1. Content Accuracy:
   - Use ONLY information explicitly stated in the context below
   - Do NOT add external knowledge or interpretations
   - If the context doesn't contain the answer, say "I don't have this information in the available documents."

2. Legal Categories to Consider:
   When answering, identify which area(s) of law apply:
   - Corporate Law (Companies Act, MCA rules, director duties, corporate governance)
   - Criminal Law (IPC sections, offenses, penalties)
   - Civil Procedure (CPC, civil litigation, suits, appeals)
   - Criminal Procedure (CrPC, FIR, investigation, arrest, bail, trial)
   - Corporate Litigation (NCLT, NCLAT, company disputes, insolvency)
   - Regulatory Compliance (SEBI, FEMA, GST, labour laws, POSH, Shops & Establishment)
   - Contract Law (agreements, breach, remedies)
   - Constitutional Law (fundamental rights, constitutional remedies)

3. Formatting for Clarity:
   - Use bullet points (-) or numbered lists (1., 2., 3.) when presenting multiple items
   - Put each distinct point on a separate line
   - Use section headings for different legal categories
   - Add line breaks between sections
   - Preserve structure from the original documents

4. Friendly & Professional Tone:
   - Explain legal concepts in clear, simple language
   - Be approachable but maintain professional accuracy
   - Help the user understand complex legal terms

Context from Legal Documents:
{context}

Question: {question}

Answer:"""
        
        self.prompt = PromptTemplate(
            template=self.prompt_template,
            input_variables=["context", "question"]
        )
        
        # ENHANCED: General Knowledge Prompt with Comprehensive Indian Legal Framework
        self.general_prompt_template = """You are a friendly and knowledgeable Indian legal assistant. Your goal is to provide accurate, helpful legal information in a way that's easy to understand.

COMPREHENSIVE LEGAL FRAMEWORK TO CONSIDER:

1. CORPORATE LAW & DUTIES:
   - Companies Act, 2013 (company formation, governance, director duties, meetings, accounts)
   - MCA (Ministry of Corporate Affairs) regulations and circulars
   - LLP Act, 2008 (Limited Liability Partnerships)
   - Director duties: fiduciary duty, duty of care, avoiding conflicts of interest
   - Corporate governance, compliance requirements, ROC filings
   - Insider trading, related party transactions

2. CRIMINAL LAW (IPC):
   - Indian Penal Code sections for offenses (theft, fraud, cheating, assault, defamation, etc.)
   - Bhartiya Nyaya Sanhita, 2023 (BNS) - new criminal code replacing IPC
   - Penalties and punishments for various offenses
   - Cognizable vs non-cognizable offenses
   - Bailable vs non-bailable offenses

3. CRIMINAL PROCEDURE (CrPC):
   - Bhartiya Nagarik Suraksha Sanhita, 2023 (BNSS) - new criminal procedure code
   - FIR filing and investigation process
   - Arrest procedures and rights of accused
   - Bail provisions (regular bail, anticipatory bail)
   - Trial procedure in criminal courts
   - Appeals and revisions

4. CIVIL PROCEDURE (CPC):
   - Civil suits and litigation process
   - Jurisdiction of civil courts
   - Pleadings, written statements, evidence
   - Interim reliefs and injunctions
   - Decree, execution, appeals
   - Alternative Dispute Resolution (Arbitration, Mediation, Conciliation)

5. CORPORATE LITIGATION:
   - NCLT (National Company Law Tribunal) proceedings
   - Company disputes, oppression and mismanagement
   - IBC (Insolvency and Bankruptcy Code) - CIRP, liquidation
   - Shareholder disputes and derivative actions
   - NCLAT appeals

6. REGULATORY COMPLIANCE:
   - SEBI (Securities and Exchange Board of India) - securities law, IPOs, insider trading
   - FEMA (Foreign Exchange Management Act) - foreign investment, repatriation
   - GST (Goods and Services Tax) - tax compliance
   - Labour Laws - Shops & Establishment Act, ESI, EPF, gratuity
   - POSH Act (Sexual Harassment at Workplace)
   - Consumer Protection Act, 2019
   - IT Act, 2000 - cyber crimes, data protection
   - Competition Act, 2002 - anti-competitive practices

7. CONTRACT LAW:
   - Indian Contract Act, 1872
   - Formation of contracts, offer and acceptance
   - Consideration, capacity, free consent
   - Breach of contract and remedies
   - Specific Relief Act - specific performance
   - Sale of Goods Act, 1930

8. CONSTITUTIONAL LAW:
   - Fundamental Rights (Articles 12-35)
   - Constitutional remedies - writs (habeas corpus, mandamus, certiorari, prohibition, quo warranto)
   - Directive Principles of State Policy
   - Judicial review

YOUR RESPONSE STRUCTURE:

1. Start with a friendly one-line summary
   Example: "Let me help you understand this legal matter clearly."

2. Identify the relevant legal category(ies)
   Example: "This falls under Criminal Law (IPC Section 420 - Cheating) and Criminal Procedure (CrPC)."

3. Explain the applicable law in simple, everyday language
   - Cite specific sections (IPC, Companies Act, CrPC, etc.)
   - Explain what the law means in plain English
   - Avoid heavy legal jargon; if you must use technical terms, define them immediately

4. Provide consequences and procedures
   - What are the penalties? (imprisonment, fines)
   - What's the procedure? (FIR, complaint, NCLT filing, civil suit, etc.)
   - What are the defenses or rights available?

5. Offer practical steps the user can take
   - How to file a complaint
   - When to consult a lawyer
   - What documents to gather
   - Time limitations (limitation periods)

6. Use bullet points for clarity
   Present main points in a bulleted or numbered list for easy reading

7. Add a helpful example if appropriate
   Use a simple, relatable scenario to illustrate the legal concept

8. End with a concise closing takeaway
   Example: "Remember, this is general legal information. For your specific situation, it's best to consult a qualified lawyer."

IMPORTANT DISCLAIMERS:
- This is general legal information, not personal legal advice
- Mention when something depends on specific conditions (e.g., "This may vary based on whether the person is a workman or managerial employee")
- Recommend consulting a lawyer for specific situations
- Be clear about jurisdictional variations (Central vs State laws)

TONE & STYLE:
- Friendly and approachable, like talking to a friend
- Patient and understanding
- Clear and concise - avoid unnecessary complexity
- Professional yet warm
- Empowering - help the user understand their legal position

USER QUESTION:
{question}

YOUR ANSWER:
"""
        
        # ENHANCED: Layman Mode Prompt - Extra Simplified
        self.layman_prompt_template = """You are a friendly legal guide helping someone who doesn't know much about law. Your job is to explain legal concepts in the simplest way possible, as if you're explaining to a friend.

ULTRA-SIMPLE EXPLANATION RULES:

1. Use everyday words - avoid legal jargon completely
   Instead of: "cognizable offense" → Say: "a serious crime where police can arrest without a warrant"
   Instead of: "fiduciary duty" → Say: "the responsibility to act in someone else's best interest"
   Instead of: "plaintiff" → Say: "the person who files the complaint"

2. Structure for maximum clarity:
   - Start with: "Here's what you need to know in simple terms..."
   - Break down complex ideas into small, digestible pieces
   - Use analogies and real-life examples
   - End with: "To sum up in one sentence..."

3. Explain like you're talking to a family member:
   - "Think of it like this..."
   - "In simple words..."
   - "Basically, what this means is..."
   - "Here's why this matters to you..."

4. Use bullet points for steps or options:
   - What can happen
   - What you should do
   - What to avoid

5. Be reassuring and practical:
   - "Don't worry, here's what this means..."
   - "The first thing you should do is..."
   - "This is common, and here's how it's usually handled..."

6. Identify the legal area simply:
   - "This is about business/company law"
   - "This is a criminal matter"
   - "This is about contracts/agreements"
   - "This deals with government rules and compliance"

7. Always end with clear next steps
   "What you should do now: [practical step-by-step guidance]"

USER QUESTION:
{question}

YOUR SIMPLE, FRIENDLY ANSWER:
"""
        
        # Chain configuration
        self.chain_type_kwargs = {"prompt": self.prompt}
        
        # Initialize QA chain only if vector_store and retriever exist
        if self.vector_store is not None and self.basic_retriever is not None:
            def format_docs(docs):
                return "\n\n".join(doc.page_content for doc in docs)
            
            self.qa_chain = (
                {
                    "context": self.basic_retriever | format_docs,
                    "question": RunnablePassthrough()
                }
                | self.prompt
                | self.llm
                | StrOutputParser()
            )
        else:
            self.qa_chain = None
            logger.warning("QA chain not initialized - waiting for documents to be processed")
        
        # Security and content filtering
        self.content_filter = ContentFilter()
        self.token_counter = TokenCounter()
        
        # Session statistics
        self.session_stats = {
            'total_queries': 0,
            'flagged_queries': 0,
            'total_tokens_used': 0,
            'input_tokens_used': 0,
            'output_tokens_used': 0,
            'start_time': time.time()
        }
        
        logger.info(f"ChatbotManager initialized with persistent collection: {self.collection_name}")
    
    def _initialize_custom_llm(
        self,
        custom_url: str,
        api_key: str,
        model_name: str,
        temperature: float,
        max_tokens: int
    ):
        """
        Initialize custom/internal LLM endpoint
        
        Supports OpenAI-compatible APIs including:
        - vLLM deployments (internal model serving)
        - FastChat servers
        - Ollama local models
        - LM Studio
        - Internal company LLMs (trained on intranet data)
        - Any OpenAI-compatible endpoint
        
        Args:
            custom_url: Base URL of the custom LLM API
            api_key: API key (optional for internal endpoints)
            model_name: Model name to use
            temperature: Temperature setting
            max_tokens: Max tokens in response
            
        Returns:
            Initialized LLM instance
        """
        try:
            from langchain_openai import ChatOpenAI
            
            logger.info(f"Connecting to custom LLM endpoint: {custom_url}")
            logger.info(f"Model: {model_name}")
            
            # Create custom LLM with OpenAI-compatible interface
            custom_llm = ChatOpenAI(
                base_url=custom_url,  # Your internal endpoint
                api_key=api_key or "not-needed",  # Some internal endpoints don't need API keys
                model=model_name,
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=60.0,
                max_retries=2,
                request_timeout=60.0
            )
            
            # Test connection with a simple query
            try:
                test_response = custom_llm.invoke("Test")
                logger.info("✅ Custom LLM connection test successful")
            except Exception as test_error:
                logger.warning(f"⚠️ Custom LLM test query failed: {test_error}")
                logger.warning("Proceeding anyway - errors may occur during actual use")
            
            return custom_llm
            
        except ImportError as e:
            error_msg = "langchain-openai not installed. Install with: pip install langchain-openai"
            logger.error(error_msg)
            raise ImportError(error_msg) from e
        except Exception as e:
            error_msg = f"Failed to initialize custom LLM: {e}"
            logger.error(error_msg)
            raise RuntimeError(error_msg) from e
    
    def _reinitialize_vector_store(self):
        """Reinitialize vector store after documents have been processed"""
        try:
            logger.info(f"Attempting to reinitialize vector store for collection: {self.collection_name}")
            
            # Check if collection exists
            collections = self.qdrant_client.get_collections().collections
            collection_exists = any(c.name == self.collection_name for c in collections)
            
            if not collection_exists:
                logger.warning(f"Collection '{self.collection_name}' still does not exist")
                return
            
            # Try newer initialization method
            try:
                self.vector_store = QdrantVectorStore(
                    client=self.qdrant_client,
                    collection_name=self.collection_name,
                    embedding=self.embeddings
                )
            except TypeError:
                # Fallback to older method
                self.vector_store = QdrantVectorStore.from_existing_collection(
                    embedding=self.embeddings,
                    collection_name=self.collection_name,
                    url=self.qdrant_url,
                    api_key=self.qdrant_api_key
                )
            
            # Reinitialize retrievers
            self.basic_retriever = self.vector_store.as_retriever(
                search_type="similarity_score_threshold",
                search_kwargs={
                    "k": self.retrieval_k,
                    "score_threshold": self.score_threshold
                }
            )
            
            self.advanced_retriever = AdvancedRetriever(
                self.vector_store,
                self.embeddings,
                k=self.retrieval_k,
                score_threshold=self.score_threshold
            )
            
            # Reinitialize QA chain
            def format_docs(docs):
                return "\n\n".join(doc.page_content for doc in docs)
            
            self.qa_chain = (
                {
                    "context": self.basic_retriever | format_docs,
                    "question": RunnablePassthrough()
                }
                | self.prompt
                | self.llm
                | StrOutputParser()
            )
            
            logger.info("✅ Vector store, retrievers, and QA chain successfully reinitialized")
            
        except Exception as e:
            logger.error(f"Failed to reinitialize vector store: {e}")
            raise
    
    def _calculate_real_tokens(self, query: str, answer: str, context: str) -> Dict[str, int]:
        """FIXED: Calculate actual token usage accurately"""
        # Input tokens: query + context + prompt overhead
        prompt_overhead = self.token_counter.count_tokens(self.prompt_template)
        query_tokens = self.token_counter.count_tokens(query)
        context_tokens = self.token_counter.count_tokens(context)
        
        input_tokens = query_tokens + context_tokens + prompt_overhead
        
        # Output tokens: the generated answer
        output_tokens = self.token_counter.count_tokens(answer)
        
        # Total tokens
        total_tokens = input_tokens + output_tokens
        
        return {
            'input_tokens': input_tokens,
            'output_tokens': output_tokens,
            'total_tokens': total_tokens,
            'query_tokens': query_tokens,
            'context_tokens': context_tokens
        }
    
    @rate_limit()
    def get_response(
        self,
        query: str,
        enable_content_filter: bool = True,
        enable_pii_detection: bool = True,
        use_rag: bool = True,  # NEW: Control whether to use RAG or direct LLM
        layman_mode: bool = False  # NEW: Simple explanation mode
    ) -> Dict[str, Any]:
        """Generate response with comprehensive security and token tracking - HYBRID MODE"""
        
        try:
            self.session_stats['total_queries'] += 1
            
            # Validate and sanitize query
            if not query or not query.strip():
                return self._create_error_response("Empty query provided", "invalid_input")
            
            query = query.strip()[:MAX_QUERY_LENGTH]
            
            # Pre-process query security
            if enable_content_filter or enable_pii_detection:
                # Check for PII
                if enable_pii_detection:
                    has_pii, pii_types, cleaned_query = self.content_filter.scan_for_pii(query)
                    if has_pii:
                        self.session_stats['flagged_queries'] += 1
                        return self._create_error_response(
                            f"Query contains PII: {', '.join(pii_types)}",
                            "pii_detected"
                        )
                
                # Check content safety
                if enable_content_filter:
                    has_safety_issues, safety_issues = self.content_filter.check_content_safety(query)
                    if has_safety_issues:
                        self.session_stats['flagged_queries'] += 1
                        return self._create_error_response(
                            f"Query blocked: {', '.join(safety_issues)}",
                            "content_filtered"
                        )
            
            # NEW: Decide between RAG and direct LLM response
            # Reinitialize vector store if needed
            if use_rag and (self.vector_store is None or self.basic_retriever is None):
                try:
                    self._reinitialize_vector_store()
                except Exception as e:
                    logger.warning(f"Could not reinitialize vector store: {e}. Falling back to direct LLM.")
                    use_rag = False
            
            if use_rag and self.vector_store is not None and self.basic_retriever is not None:
                # Try RAG first (document-based answer)
                return self._get_rag_response(query, enable_content_filter, enable_pii_detection)
            else:
                # Direct LLM response (general knowledge) - with layman mode option
                return self._get_direct_llm_response(query, enable_content_filter, enable_pii_detection, layman_mode)
            
        except Exception as e:
            logger.error(f"Response generation error: {e}")
            return self._create_error_response(f"System error: {str(e)}", "system_error")
    
    def _get_rag_response(
        self,
        query: str,
        enable_content_filter: bool,
        enable_pii_detection: bool
    ) -> Dict[str, Any]:
        """Get response using RAG (Retrieval-Augmented Generation)"""
        try:
            # Step 1: Use advanced retrieval for better results
            relevant_docs = self.advanced_retriever.hybrid_search(query, k=self.retrieval_k)
            
            # Step 2: Check if we found relevant documents
            if not relevant_docs or len(relevant_docs) == 0:
                # No relevant docs found - return "don't know" response
                logger.info(f"No relevant documents found for query: {query}")
                return {
                    "answer": "I don't have enough information in the available documents to answer this question.",
                    "sources": [],
                    "is_flagged": False,
                    "flag_reason": None,
                    "tokens_used": 50,  # Minimal tokens for this response
                    "input_tokens": 30,
                    "output_tokens": 20,
                    "processing_time": time.time(),
                    "response_type": "rag"
                }
            
            # Step 3: Rerank documents for better relevance
            reranked_docs = self.advanced_retriever.rerank_documents(query, relevant_docs)
            
            # Step 4: Build context from reranked documents
            context_parts = []
            for doc in reranked_docs:
                doc_info = f"Document: {doc.metadata.get('file_name', 'Unknown')} (Page {doc.metadata.get('page', 'N/A')})"
                content = f"Content: {doc.page_content}"
                context_parts.append(f"{doc_info}\n{content}")
            
            context = "\n\n".join(context_parts)
            
            # Create the prompt for RAG
            formatted_prompt = self.prompt.format(context=context, question=query.strip())
            
            # Get response from LLM
            response = self.llm.invoke(formatted_prompt)
            answer = response.content if hasattr(response, 'content') else str(response)
            source_documents = reranked_docs
            
            # FIXED: Calculate real token usage
            token_breakdown = self._calculate_real_tokens(query, answer, context)
            
            # Post-process response security
            if enable_content_filter or enable_pii_detection:
                filtered_answer, has_filter_issues, filter_issues = self.content_filter.filter_response(answer)
                if has_filter_issues:
                    self.session_stats['flagged_queries'] += 1
                    return {
                        "answer": filtered_answer,
                        "sources": self._process_sources(source_documents),
                        "is_flagged": True,
                        "flag_reason": f"Response filtered: {', '.join(filter_issues)}",
                        "tokens_used": token_breakdown['total_tokens'],
                        "input_tokens": token_breakdown['input_tokens'],
                        "output_tokens": token_breakdown['output_tokens'],
                        "response_type": "rag"
                    }
                answer = filtered_answer
            
            # FIXED: Update session stats with real token usage
            self.session_stats['total_tokens_used'] += token_breakdown['total_tokens']
            self.session_stats['input_tokens_used'] += token_breakdown['input_tokens']
            self.session_stats['output_tokens_used'] += token_breakdown['output_tokens']
            
            # Process sources
            sources = self._process_sources(source_documents)
            
            return {
                "answer": answer.strip() or "I couldn't find relevant information to answer your question based on the available documents.",
                "sources": sources,
                "is_flagged": False,
                "flag_reason": None,
                "tokens_used": token_breakdown['total_tokens'],
                "input_tokens": token_breakdown['input_tokens'],
                "output_tokens": token_breakdown['output_tokens'],
                "processing_time": time.time(),
                "response_type": "rag"
            }
            
        except Exception as e:
            logger.error(f"RAG response error: {e}")
            return self._create_error_response(
                "Unable to process documents to answer your question. Please try rephrasing your question.",
                "rag_processing_error"
            )
    
    def _get_direct_llm_response(
        self,
        query: str,
        enable_content_filter: bool,
        enable_pii_detection: bool,
        layman_mode: bool = False
    ) -> Dict[str, Any]:
        """Get response directly from LLM without RAG (general knowledge) with optional layman mode"""
        try:
            # Choose appropriate prompt based on mode
            if layman_mode:
                general_prompt = self.layman_prompt_template.format(question=query)
                response_type = "layman"
            else:
                general_prompt = self.general_prompt_template.format(question=query)
                response_type = "general_knowledge"
            
            # Get response from LLM
            response = self.llm.invoke(general_prompt)
            answer = response.content if hasattr(response, 'content') else str(response)
            
            # Calculate token usage (no document context)
            token_breakdown = self._calculate_real_tokens(query, answer, general_prompt)
            
            # Post-process response security
            if enable_content_filter or enable_pii_detection:
                filtered_answer, has_filter_issues, filter_issues = self.content_filter.filter_response(answer)
                if has_filter_issues:
                    self.session_stats['flagged_queries'] += 1
                    return {
                        "answer": filtered_answer,
                        "sources": [],
                        "is_flagged": True,
                        "flag_reason": f"Response filtered: {', '.join(filter_issues)}",
                        "tokens_used": token_breakdown['total_tokens'],
                        "input_tokens": token_breakdown['input_tokens'],
                        "output_tokens": token_breakdown['output_tokens'],
                        "response_type": response_type
                    }
                answer = filtered_answer
            
            # Update session stats
            self.session_stats['total_tokens_used'] += token_breakdown['total_tokens']
            self.session_stats['input_tokens_used'] += token_breakdown['input_tokens']
            self.session_stats['output_tokens_used'] += token_breakdown['output_tokens']
            
            return {
                "answer": answer.strip() or "I'm not sure how to answer that question.",
                "sources": [],  # No document sources for general knowledge
                "is_flagged": False,
                "flag_reason": None,
                "tokens_used": token_breakdown['total_tokens'],
                "input_tokens": token_breakdown['input_tokens'],
                "output_tokens": token_breakdown['output_tokens'],
                "processing_time": time.time(),
                "response_type": response_type
            }
            
        except Exception as e:
            logger.error(f"General knowledge response error: {e}")
            return self._create_error_response(f"System error: {str(e)}", "system_error")

    def _create_error_response(self, message: str, reason: str) -> Dict[str, Any]:
        """Create standardized error response"""
        return {
            "answer": f"⚠️ {message}",
            "sources": [],
            "is_flagged": True,
            "flag_reason": reason,
            "tokens_used": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "processing_time": time.time(),
            "response_type": "error"
        }

    def _process_sources(self, source_documents: List[Document]) -> List[Dict[str, Any]]:
        """Process and secure source documents - simplified for UI"""
        if not source_documents:
            return []
        
        sources = []
        seen_sources = set()
        
        for i, doc in enumerate(source_documents[:8]):  # Limit to 8 sources
            try:
                metadata = doc.metadata
                
                # Extract metadata safely
                file_name = str(metadata.get('file_name', metadata.get('source', 'Unknown Document')))
                if '/' in file_name or '\\' in file_name:
                    file_name = os.path.basename(file_name)
                file_name = file_name[:50]  # Limit length
                
                page = metadata.get('page', 'N/A')
                
                # Deduplication based on file and page
                source_key = f"{file_name}_{page}"
                if source_key in seen_sources:
                    continue
                
                source_info = {
                    "file_name": file_name,
                    "page": page
                }
                
                sources.append(source_info)
                seen_sources.add(source_key)
                
            except Exception as e:
                logger.warning(f"Error processing source {i}: {e}")
                continue
        
        return sources

    def cleanup_collection(self) -> None:
        """
        FIXED: This method should NOT delete the persistent collection
        The collection "Legal_documents" is shared across all sessions
        Only call this if you want to delete ALL documents permanently
        """
        try:
            logger.warning(f"⚠️ cleanup_collection() called - this will delete the entire '{self.collection_name}' collection!")
            logger.warning("This is a shared persistent collection. Deletion will affect all users/sessions.")
            # Commenting out the actual deletion to prevent accidental data loss
            # If you really want to clear the collection, uncomment the line below:
            # self.qdrant_client.delete_collection(self.collection_name)
            logger.info(f"Collection cleanup skipped to preserve persistent data: {self.collection_name}")
        except Exception as e:
            logger.warning(f"Collection cleanup operation: {e}")

    def get_session_stats(self) -> Dict[str, Any]:
        """Get current session statistics with accurate token data"""
        current_time = time.time()
        session_duration = current_time - self.session_stats['start_time']
        
        return {
            **self.session_stats,
            "session_duration_seconds": round(session_duration, 2),
            "average_tokens_per_query": (
                self.session_stats['total_tokens_used'] / max(1, self.session_stats['total_queries'])
            ),
            "flagged_percentage": (
                (self.session_stats['flagged_queries'] / max(1, self.session_stats['total_queries'])) * 100
            ),
            "tokens_remaining": MAX_TOKENS_PER_SESSION - self.session_stats['total_tokens_used'],
            "input_output_ratio": (
                self.session_stats['input_tokens_used'] / max(1, self.session_stats['output_tokens_used'])
            )
        }

    def update_retrieval_settings(self, k: int = None, threshold: float = None) -> None:
        """Update retrieval parameters with validation"""
        try:
            if k is not None:
                self.retrieval_k = max(3, min(k, 15))
                self.advanced_retriever.k = self.retrieval_k
            
            if threshold is not None:
                self.score_threshold = max(0.0, min(threshold, 1.0))
                self.advanced_retriever.score_threshold = self.score_threshold
            
            # Update basic retriever
            self.basic_retriever = self.vector_store.as_retriever(
                search_type="similarity_score_threshold",
                search_kwargs={
                    "k": self.retrieval_k,
                    "score_threshold": self.score_threshold
                }
            )
            
            # Update chain using modern LangChain 1.0+ approach
            def format_docs(docs):
                return "\n\n".join(doc.page_content for doc in docs)
            
            self.qa_chain = (
                {
                    "context": self.basic_retriever | format_docs,
                    "question": RunnablePassthrough()
                }
                | self.prompt
                | self.llm
                | StrOutputParser()
            )
            
            logger.info(f"Updated retrieval: k={self.retrieval_k}, threshold={self.score_threshold}")
            
        except Exception as e:
            logger.error(f"Failed to update retrieval settings: {e}")

    def reset_session(self) -> None:
        """Reset session statistics"""
        self.session_stats = {
            'total_queries': 0,
            'flagged_queries': 0,
            'total_tokens_used': 0,
            'input_tokens_used': 0,
            'output_tokens_used': 0,
            'start_time': time.time()
        }
        logger.info("Session statistics reset")