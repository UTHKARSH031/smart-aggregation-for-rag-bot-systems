"""
Document Chunking Strategies
=============================

Implements 4 chunking strategies:
1. Fixed-128 tokens
2. Fixed-256 tokens  
3. Fixed-512 tokens
4. Element-based (structure-aware)
"""

import re
import warnings
from typing import List
from dataclasses import dataclass

@dataclass
class ChunkMetadata:
    """Metadata for a chunk"""
    start_char: int
    end_char: int
    element_type: str = None                                                          

class BaseChunker:
    """Fixed-size chunking strategies"""
    
    def __init__(self, chunk_size: int, overlap: int = 0):
        """
        Args:
            chunk_size: Number of tokens per chunk (128, 256, or 512)
            overlap: Number of overlapping tokens between chunks
        
        Raises:
            ValueError: If overlap >= chunk_size (would cause infinite loop)
        """
        if overlap >= chunk_size:
            raise ValueError(
                f"overlap ({overlap}) must be less than chunk_size ({chunk_size}), "
                f"otherwise chunking will never advance."
            )
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.strategy_name = f"fixed-{chunk_size}"
    
    def chunk(self, text: str, doc_id: str) -> List[dict]:
        """
        Create fixed-size chunks
        
        Process:
        1. Split text into tokens (words)
        2. Group into chunks of size N
        3. Optional: Add overlap between chunks
        """
                                                   
        tokens = text.split()
        
        chunks = []
        chunk_id = 0
        
        i = 0
        while i < len(tokens):
                                    
            chunk_tokens = tokens[i:i + self.chunk_size]
            
            if len(chunk_tokens) < 10:
                dropped_words = len(chunk_tokens)
                warnings.warn(
                    f"[{self.strategy_name}] Dropped trailing fragment of "
                    f"{dropped_words} words from '{doc_id}' (below 10-token minimum)."
                )
                break
            
            chunk_text = ' '.join(chunk_tokens)
            
            chunks.append({
                'chunk_id': f"{doc_id}_{self.strategy_name}_chunk{chunk_id}",
                'text': chunk_text,
                'doc_id': doc_id,
                'strategy': self.strategy_name,
                'tokens': len(chunk_tokens)
            })
            
            i += self.chunk_size - self.overlap
            chunk_id += 1
        
        return chunks

class ElementChunker:
    """
    Element-based chunking - Structure-aware
    
    This is what the original paper proposed!
    Instead of blindly cutting every N tokens, it:
    - Detects document structure (titles, paragraphs, tables)
    - Keeps elements together
    - Never splits tables or sections
    """
    
    def __init__(self, max_chunk_size: int = 2048, merge_small: bool = True):
        """
        Args:
            max_chunk_size: Maximum tokens per chunk
            merge_small: Whether to merge small elements together
        """
        self.max_chunk_size = max_chunk_size
        self.merge_small = merge_small
        self.strategy_name = "element-based"
    
    def chunk(self, text: str, doc_id: str) -> List[dict]:
        """
        Create structure-aware chunks
        
        Process:
        1. Detect structural elements (titles, paragraphs, tables, lists)
        2. Keep each element intact
        3. Merge small elements if needed
        4. Split large elements if they exceed max_size
        """
        elements = self._detect_elements(text)
        chunks = self._create_chunks_from_elements(elements, doc_id)
        return chunks
    
    def _detect_elements(self, text: str) -> List[dict]:
        """
        Detect structural elements in the document
        
        Detection rules:
        - Title: Lines in ALL CAPS or starting with #
        - Table: Lines with multiple "|" symbols
        - List: Lines starting with "-", "-", or numbers
        - Paragraph: Consecutive lines of regular text
        """
        lines = text.split('\n')
        elements = []
        current_element = {'type': None, 'lines': []}
        
        for line in lines:
            line_stripped = line.strip()
            
            if not line_stripped:
                                                     
                if current_element['lines']:
                    elements.append(current_element)
                    current_element = {'type': None, 'lines': []}
                continue
            
            element_type = self._classify_line(line_stripped)
            
            if current_element['type'] != element_type:
                if current_element['lines']:
                    elements.append(current_element)
                current_element = {'type': element_type, 'lines': [line]}
            else:
                current_element['lines'].append(line)
        
        if current_element['lines']:
            elements.append(current_element)
        
        return elements
    
    def _classify_line(self, line: str) -> str:
        """Classify what type of element this line is"""
                        
        if line.isupper() or line.startswith('#'):
            return 'title'
        
        if line.count('|') >= 2:
            return 'table'
        
        if re.match(r'^[\-\-\*]\s', line) or re.match(r'^\d+[\.\)]\s', line):
            return 'list'
        
        return 'paragraph'
    
    def _create_chunks_from_elements(self, elements: List[dict], doc_id: str) -> List[dict]:
        """
        Convert detected elements into chunks
        
        Rules:
        - Keep tables whole (never split!)
        - Merge small elements (< 100 tokens)
        - Split large elements (> max_chunk_size)
        """
        chunks = []
        chunk_id = 0
        
        i = 0
        while i < len(elements):
            element = elements[i]
            element_text = '\n'.join(element['lines'])
            element_tokens = len(element_text.split())
            
            if element['type'] == 'table':
                chunks.append({
                    'chunk_id': f"{doc_id}_{self.strategy_name}_chunk{chunk_id}",
                    'text': element_text,
                    'doc_id': doc_id,
                    'strategy': self.strategy_name,
                    'tokens': element_tokens,
                    'element_type': 'table'
                })
                chunk_id += 1
                i += 1
                continue
            
            if self.merge_small and element_tokens < 100 and i < len(elements) - 1:
                                                
                merged_lines = element['lines'].copy()
                j = i + 1
                merged_tokens = element_tokens
                
                while j < len(elements) and merged_tokens < self.max_chunk_size:
                    next_elem = elements[j]
                    next_tokens = len(' '.join(next_elem['lines']).split())
                    
                    if next_elem['type'] == 'table':
                        break                      
                    
                    if merged_tokens + next_tokens > self.max_chunk_size:
                        break
                    
                    merged_lines.extend(next_elem['lines'])
                    merged_tokens += next_tokens
                    j += 1
                
                chunks.append({
                    'chunk_id': f"{doc_id}_{self.strategy_name}_chunk{chunk_id}",
                    'text': '\n'.join(merged_lines),
                    'doc_id': doc_id,
                    'strategy': self.strategy_name,
                    'tokens': merged_tokens,
                    'element_type': 'merged'
                })
                chunk_id += 1
                i = j
            else:
                                                     
                chunks.append({
                    'chunk_id': f"{doc_id}_{self.strategy_name}_chunk{chunk_id}",
                    'text': element_text,
                    'doc_id': doc_id,
                    'strategy': self.strategy_name,
                    'tokens': element_tokens,
                    'element_type': element['type']
                })
                chunk_id += 1
                i += 1
        
        return chunks
