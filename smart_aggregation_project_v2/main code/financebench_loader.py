"""
FinanceBench Dataset Loader
============================

Loads real FinanceBench financial documents and questions.
Supports both PDF extraction and cached text loading.

FinanceBench: https://github.com/patronus-ai/financebench
Paper: Islam et al., "FinanceBench: A New Benchmark for Financial Question Answering" (2023)
"""

import os
import pandas as pd
import json
from typing import Dict, List, Optional

try:
    import pdfplumber
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False
    print("WARN  Warning: pdfplumber not installed. Install with: pip install pdfplumber")

class FinanceBenchLoader:
    """
    Loader for FinanceBench dataset
    
    Handles:
    - PDF text extraction
    - Caching for faster re-runs
    - CSV question loading
    - Data filtering and preprocessing
    """
    
    def __init__(self, financebench_path: str, use_cache: bool = True):
        """
        Initialize FinanceBench loader
        
        Args:
            financebench_path: Path to financebench folder
            use_cache: Whether to cache extracted PDF text
        """
        self.financebench_path = financebench_path
        self.use_cache = use_cache
        
        self.pdf_folder = os.path.join(financebench_path, 'pdfs')
        self.jsonl_path = os.path.join(financebench_path, 'data', 'financebench_open_source.jsonl')
        
        if not os.path.exists(self.pdf_folder):
            raise FileNotFoundError(
                f"PDF folder not found: {self.pdf_folder}\n"
                f"Did you download FinanceBench? See README.md for instructions."
            )
        
        if not os.path.exists(self.jsonl_path):
            raise FileNotFoundError(
                f"JSONL file not found: {self.jsonl_path}\n"
                f"Make sure FinanceBench is downloaded correctly."
            )
        
        self.cache_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'data', 'cache'
        )
        os.makedirs(self.cache_dir, exist_ok=True)
    
    def load_documents(self, max_docs: Optional[int] = None) -> Dict[str, str]:
        """
        Load FinanceBench PDF documents
        
        Args:
            max_docs: Maximum number of documents to load (None = all)
            
        Returns:
            Dictionary {doc_id: full_text}
        """
        pdf_files = sorted([f for f in os.listdir(self.pdf_folder) if f.endswith('.pdf')])
        
        if max_docs:
            pdf_files = pdf_files[:max_docs]
        
        print(f"\n{'='*70}")
        print(f"LOADING FINANCEBENCH DOCUMENTS")
        print(f"{'='*70}")
        print(f"Documents to load: {len(pdf_files)}")
        print(f"Cache enabled: {self.use_cache}")
        print(f"{'='*70}\n")
        
        documents = {}
        
        for i, pdf_file in enumerate(pdf_files, 1):
            pdf_path = os.path.join(self.pdf_folder, pdf_file)
            doc_id = pdf_file.replace('.pdf', '')
            
            print(f"[{i:2d}/{len(pdf_files):2d}] {pdf_file:50s} ", end='', flush=True)
            
            try:
                                 
                if self.use_cache:
                    text = self._load_from_cache(doc_id)
                    if text:
                        word_count = len(text.split())
                        print(f"[OK] (cached, {word_count:,} words)")
                        documents[doc_id] = text
                        continue
                
                if not PDF_AVAILABLE:
                    print("[X] (pdfplumber not installed)")
                    continue
                
                text = self._extract_text_from_pdf(pdf_path)
                word_count = len(text.split())
                
                if self.use_cache:
                    self._save_to_cache(doc_id, text)
                
                documents[doc_id] = text
                print(f"[OK] (extracted, {word_count:,} words)")
                
            except Exception as e:
                print(f"[X] Error: {str(e)[:50]}")
                continue
        
        print(f"\n{'='*70}")
        print(f"[OK] Successfully loaded {len(documents)}/{len(pdf_files)} documents")
        print(f"{'='*70}\n")
        
        return documents
    
    def load_questions(self, max_questions: Optional[int] = None, 
                      filter_doc_ids: Optional[set] = None) -> List[Dict]:
        """
        Load FinanceBench questions from CSV
        
        Args:
            max_questions: Maximum questions to load (None = all)
            filter_doc_ids: Only load questions for these documents
            
        Returns:
            List of question dictionaries
        """
        print(f"Loading questions from JSONL...")
        
        df = pd.read_json(self.jsonl_path, lines=True)
        
        questions = []
        
        for idx, row in df.iterrows():
                                                               
            doc_id = str(row.get('doc_name', row.get('document', ''))).replace('.pdf', '')
            
            if filter_doc_ids and doc_id not in filter_doc_ids:
                continue
            
            question_dict = {
                'question': str(row.get('question', '')),
                'answer': str(row.get('answer', '')),
                'doc_id': doc_id,
                'type': str(row.get('question_type', 'unknown')),
                'evidence': str(row.get('evidence', '')),
                'page': int(row.get('page', -1)) if pd.notna(row.get('page')) else -1
            }
            
            questions.append(question_dict)
            
            if max_questions and len(questions) >= max_questions:
                break
        
        print(f"[OK] Loaded {len(questions)} questions")
        
        return questions
    
    def load_data(self, max_docs: Optional[int] = None, 
                  max_questions: Optional[int] = None) -> Dict:
        """
        Load complete FinanceBench dataset using a question-first strategy.
        
        This ensures we always get the requested number of questions by first
        scanning the JSONL to find which documents have questions, then loading
        only those documents.
        
        Args:
            max_docs: Maximum documents (None = all)
            max_questions: Maximum questions (None = all 150)
            
        Returns:
            Dictionary with 'documents' and 'questions' keys
        """
        # Step 1: Load ALL questions first (no doc filter) to see what's available
        all_questions = self.load_questions(max_questions=None, filter_doc_ids=None)
        
        # Step 2: Figure out which documents have questions, ordered by question count
        from collections import Counter
        doc_question_counts = Counter(q['doc_id'] for q in all_questions)
        
        # Step 3: Check which of those documents actually exist as PDFs
        pdf_files = set(
            f.replace('.pdf', '') 
            for f in os.listdir(self.pdf_folder) if f.endswith('.pdf')
        )
        
        # Prioritize docs with the most questions to maximize question coverage
        available_docs_with_questions = [
            doc_id for doc_id, _ in doc_question_counts.most_common()
            if doc_id in pdf_files
        ]
        
        # Step 4: Select documents to load - pick enough to cover max_questions
        if max_questions:
            # Greedily select docs until we have enough questions
            selected_doc_ids = []
            question_count = 0
            for doc_id in available_docs_with_questions:
                selected_doc_ids.append(doc_id)
                question_count += doc_question_counts[doc_id]
                if question_count >= max_questions:
                    break
                if max_docs and len(selected_doc_ids) >= max_docs:
                    break
            
            # If we still haven't reached max_docs, we can add more
            if max_docs and len(selected_doc_ids) < max_docs:
                for doc_id in available_docs_with_questions:
                    if doc_id not in selected_doc_ids:
                        selected_doc_ids.append(doc_id)
                        if len(selected_doc_ids) >= max_docs:
                            break
        else:
            selected_doc_ids = available_docs_with_questions
            if max_docs:
                selected_doc_ids = selected_doc_ids[:max_docs]
        
        # Step 5: Load only the selected documents
        selected_doc_set = set(selected_doc_ids)
        documents = self._load_selected_documents(selected_doc_set)
        
        # Step 6: Filter questions to only loaded documents and apply max_questions
        loaded_doc_ids = set(documents.keys())
        questions = [q for q in all_questions if q['doc_id'] in loaded_doc_ids]
        if max_questions:
            questions = questions[:max_questions]
        
        print(f"\n{'='*70}")
        print(f"FINANCEBENCH DATASET LOADED (question-first strategy)")
        print(f"{'='*70}")
        print(f"Documents:  {len(documents)}")
        print(f"Questions:  {len(questions)}")
        print(f"Total words: {sum(len(text.split()) for text in documents.values()):,}")
        print(f"{'='*70}\n")
        
        return {
            'documents': documents,
            'questions': questions
        }
    
    def _load_selected_documents(self, doc_ids: set) -> Dict[str, str]:
        """
        Load specific documents by their IDs
        
        Args:
            doc_ids: Set of document IDs to load
            
        Returns:
            Dictionary {doc_id: full_text}
        """
        print(f"\n{'='*70}")
        print(f"LOADING FINANCEBENCH DOCUMENTS (selected)")
        print(f"{'='*70}")
        print(f"Documents to load: {len(doc_ids)}")
        print(f"Cache enabled: {self.use_cache}")
        print(f"{'='*70}\n")
        
        documents = {}
        sorted_ids = sorted(doc_ids)
        
        for i, doc_id in enumerate(sorted_ids, 1):
            pdf_file = f"{doc_id}.pdf"
            pdf_path = os.path.join(self.pdf_folder, pdf_file)
            
            print(f"[{i:2d}/{len(sorted_ids):2d}] {pdf_file:50s} ", end='', flush=True)
            
            try:
                if self.use_cache:
                    text = self._load_from_cache(doc_id)
                    if text:
                        word_count = len(text.split())
                        print(f"[OK] (cached, {word_count:,} words)")
                        documents[doc_id] = text
                        continue
                
                if not PDF_AVAILABLE:
                    print("[X] (pdfplumber not installed)")
                    continue
                
                if not os.path.exists(pdf_path):
                    print("[X] (PDF not found)")
                    continue
                
                text = self._extract_text_from_pdf(pdf_path)
                word_count = len(text.split())
                
                if self.use_cache:
                    self._save_to_cache(doc_id, text)
                
                documents[doc_id] = text
                print(f"[OK] (extracted, {word_count:,} words)")
                
            except Exception as e:
                print(f"[X] Error: {str(e)[:50]}")
                continue
        
        print(f"\n{'='*70}")
        print(f"[OK] Successfully loaded {len(documents)}/{len(doc_ids)} documents")
        print(f"{'='*70}\n")
        
        return documents
    
    def _extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract text from PDF using pdfplumber"""
        text_parts = []
        
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    text_parts.append(text)
        
        return '\n\n'.join(text_parts)
    
    def _load_from_cache(self, doc_id: str) -> Optional[str]:
        """Load extracted text from cache"""
        cache_file = os.path.join(self.cache_dir, f"{doc_id}.txt")
        
        if os.path.exists(cache_file):
            with open(cache_file, 'r', encoding='utf-8') as f:
                return f.read()
        return None
    
    def _save_to_cache(self, doc_id: str, text: str):
        """Save extracted text to cache"""
        cache_file = os.path.join(self.cache_dir, f"{doc_id}.txt")
        
        with open(cache_file, 'w', encoding='utf-8') as f:
            f.write(text)

def get_financebench_data(
    financebench_path: str,
    max_docs: Optional[int] = 10,
    max_questions: Optional[int] = 20,
    use_cache: bool = True
) -> Dict:
    """
    Convenience function to load FinanceBench data
    
    Args:
        financebench_path: Path to financebench folder
        max_docs: Max documents to load (None = all 80)
        max_questions: Max questions to load (None = all 150)
        use_cache: Whether to cache extracted PDFs
        
    Returns:
        Dictionary with 'documents' and 'questions'
        
    Example:
        >>> data = get_financebench_data(
        ...     financebench_path="C:/Users/name/Downloads/financebench",
        ...     max_docs=5,
        ...     max_questions=10
        ... )
        >>> print(f"Loaded {len(data['documents'])} docs")
    """
    loader = FinanceBenchLoader(financebench_path, use_cache=use_cache)
    return loader.load_data(max_docs, max_questions)

if __name__ == "__main__":
                  
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python financebench_loader.py <path_to_financebench>")
        sys.exit(1)
    
    financebench_path = sys.argv[1]
    
    print("Testing FinanceBench loader...")
    data = get_financebench_data(
        financebench_path=financebench_path,
        max_docs=3,
        max_questions=5
    )
    
    print("\nTest successful!")
    print(f"Loaded {len(data['documents'])} documents")
    print(f"Loaded {len(data['questions'])} questions")
