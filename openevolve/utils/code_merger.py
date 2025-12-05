"""
Code Merger Utility for EVOLVE-BLOCK handling

This module provides utilities to:
1. Split code into pre-block, evolve-block, and post-block sections
2. Merge LLM-generated code back with the preserved sections

This ensures that code outside EVOLVE-BLOCK (like `run_packing()`) is NEVER modified.
"""

import re
import logging
from typing import Tuple, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class CodeSections:
    """Represents the three sections of code split by EVOLVE-BLOCK markers"""
    pre_block: str      # Everything before EVOLVE-BLOCK-START
    evolve_block: str   # Content inside the EVOLVE-BLOCK markers
    post_block: str     # Everything after EVOLVE-BLOCK-END
    
    has_markers: bool = True  # Whether the original code had EVOLVE-BLOCK markers


def split_code(code: str) -> CodeSections:
    """
    Split code into pre-block, evolve-block, and post-block sections.
    
    Args:
        code: The full program code
        
    Returns:
        CodeSections with the three parts separated
    """
    lines = code.split('\n')
    
    pre_block_lines = []
    evolve_block_lines = []
    post_block_lines = []
    
    in_evolve_block = False
    found_start = False
    found_end = False
    
    for line in lines:
        if '# EVOLVE-BLOCK-START' in line or '#EVOLVE-BLOCK-START' in line:
            found_start = True
            in_evolve_block = True
            pre_block_lines.append(line)  # Keep the marker in pre_block
            continue
        elif '# EVOLVE-BLOCK-END' in line or '#EVOLVE-BLOCK-END' in line:
            found_end = True
            in_evolve_block = False
            post_block_lines.append(line)  # Keep the marker in post_block
            continue
        
        if not found_start:
            pre_block_lines.append(line)
        elif in_evolve_block:
            evolve_block_lines.append(line)
        else:  # After EVOLVE-BLOCK-END
            post_block_lines.append(line)
    
    has_markers = found_start and found_end
    
    if not has_markers:
        logger.warning("No EVOLVE-BLOCK markers found in code")
        # Return the whole code as evolve_block if no markers
        return CodeSections(
            pre_block="",
            evolve_block=code,
            post_block="",
            has_markers=False,
        )
    
    return CodeSections(
        pre_block='\n'.join(pre_block_lines),
        evolve_block='\n'.join(evolve_block_lines),
        post_block='\n'.join(post_block_lines),
        has_markers=True,
    )


def merge_code(sections: CodeSections, new_evolve_block: str) -> str:
    """
    Merge the preserved sections with new evolve block content.
    
    Args:
        sections: The original CodeSections
        new_evolve_block: The LLM-generated content for inside the EVOLVE-BLOCK
        
    Returns:
        Complete merged code
    """
    if not sections.has_markers:
        # No markers - just return the new code as-is
        return new_evolve_block
    
    # Clean up the new evolve block
    new_evolve_block = new_evolve_block.strip()
    
    # Remove any EVOLVE-BLOCK markers that the LLM might have included
    new_evolve_block = re.sub(r'#\s*EVOLVE-BLOCK-START\s*\n?', '', new_evolve_block)
    new_evolve_block = re.sub(r'#\s*EVOLVE-BLOCK-END\s*\n?', '', new_evolve_block)
    
    # Build the merged code
    parts = []
    
    if sections.pre_block:
        parts.append(sections.pre_block)
    
    # Add the new evolve block content
    parts.append(new_evolve_block)
    
    if sections.post_block:
        parts.append(sections.post_block)
    
    return '\n'.join(parts)


def extract_evolve_block_from_response(response: str, language: str = "python") -> Optional[str]:
    """
    Extract the evolve block content from an LLM response.
    
    The LLM might output:
    - Just the code
    - Code wrapped in ```python ... ```
    - Code with EVOLVE-BLOCK markers
    
    Args:
        response: The LLM's response
        language: The programming language
        
    Returns:
        The extracted code content, or None if extraction failed
    """
    if not response:
        return None
    
    # Try to extract from code blocks first
    code_block_pattern = rf'```(?:{language})?\s*\n?(.*?)```'
    matches = re.findall(code_block_pattern, response, re.DOTALL | re.IGNORECASE)
    
    if matches:
        # Use the first (or largest) code block
        code = max(matches, key=len).strip()
    else:
        # No code blocks - use the response as-is
        code = response.strip()
    
    # Remove EVOLVE-BLOCK markers if present (LLM might include them)
    code = re.sub(r'#\s*EVOLVE-BLOCK-START\s*\n?', '', code)
    code = re.sub(r'#\s*EVOLVE-BLOCK-END\s*\n?', '', code)
    
    # Also remove any leading/trailing markers that might be on their own lines
    lines = code.split('\n')
    cleaned_lines = []
    for line in lines:
        if 'EVOLVE-BLOCK-START' in line or 'EVOLVE-BLOCK-END' in line:
            continue
        cleaned_lines.append(line)
    
    return '\n'.join(cleaned_lines).strip()

