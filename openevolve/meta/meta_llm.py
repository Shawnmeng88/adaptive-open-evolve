"""
Meta-LLM for Seed Prompt Generation and Refinement

The Meta-LLM is responsible for:
1. Generating initial seed prompts based on problem description
2. Refining seed prompts based on convergence feedback
3. Learning patterns that lead to faster convergence
"""

import logging
import os
from typing import Optional

from openai import AsyncOpenAI

from openevolve.meta.seed_prompt import SeedPromptHistory

logger = logging.getLogger(__name__)


# Template for generating initial seed prompts
INITIAL_SEED_PROMPT_TEMPLATE = """You are a meta-prompt engineer. Your task is to create an effective system message (seed prompt) that will guide an LLM to evolve code solutions through iterative improvement.

## Problem Description
{problem_description}

## Initial Code
```
{initial_code}
```

## Evaluation Criteria
{evaluation_criteria}

## CRITICAL OUTPUT FORMAT
The code has EVOLVE-BLOCK markers. The LLM you are prompting must:
- Output ONLY the code that goes BETWEEN the `# EVOLVE-BLOCK-START` and `# EVOLVE-BLOCK-END` markers
- NOT include the markers themselves in output
- NOT output any code outside the markers (like `run_packing()` - those are preserved automatically)
- The system merges the LLM output with preserved code sections

Include this instruction clearly in the system message you create.

## Your Task
Create a comprehensive system message that will help an LLM generate increasingly better code solutions. The system message should:

1. **Define the role clearly**: What kind of expert should the LLM act as?
2. **Explain the optimization goal**: What metrics should be improved?
3. **Provide domain knowledge**: What techniques, algorithms, or approaches might be relevant?
4. **Set output format**: Tell the LLM to output ONLY the evolve block content
5. **Guide exploration**: Encourage trying different approaches while respecting constraints
6. **Warn about common pitfalls**: What mistakes should the LLM avoid?

The prompt should be detailed enough to guide the LLM effectively but not so long that it overwhelms the context.

Output ONLY the system message, no explanations or preamble:"""


# Template for refining seed prompts based on feedback
REFINE_SEED_PROMPT_TEMPLATE = """You are a meta-prompt engineer improving the system message for code evolution.

## Problem
{problem_description}

## Best Result So Far
Score: {best_final_score}
Key approach in best code:
{best_code_summary}

## What Worked (keep these in the new prompt)
{successful_patterns}

## CRITICAL: What Failed (the new prompt MUST warn against these)
{stuck_patterns}

## Current Best Prompt (to improve upon)
{current_best_prompt}

## MANDATORY OUTPUT FORMAT INSTRUCTION
The system message MUST tell the LLM to:
- Output ONLY the code between `# EVOLVE-BLOCK-START` and `# EVOLVE-BLOCK-END` markers
- NOT include the markers or any code outside them
- The outer code (like `run_packing()`) is preserved automatically by the system

## Your Task
Write an improved system message that:
1. Builds on the successful approach described above
2. EXPLICITLY FORBIDS the failed approaches listed above
3. INCLUDES the mandatory output format instruction above
4. Suggests specific alternative strategies to try
5. Is concrete and domain-specific (not generic advice)

Output ONLY the improved system message:"""


class MetaLLM:
    """
    LLM that generates and refines seed prompts for the inner evolution loop
    
    This is the "outer loop" intelligence that learns from evolution results
    and produces better prompts over time.
    """
    
    def __init__(
        self,
        model: str = "meta-llama/Meta-Llama-3.1-70B-Instruct",
        temperature: float = 0.7,
        api_base: Optional[str] = None,
        api_key: Optional[str] = None,
    ):
        """
        Args:
            model: Model to use for meta-prompting
            temperature: Temperature for generation
            api_base: API base URL (defaults to OpenAI)
            api_key: API key (defaults to OPENAI_API_KEY env var)
        """
        self.model = model
        self.temperature = temperature
        
        # Use provided values or fall back to environment
        self.api_base = api_base
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        
        if self.api_base:
            self.client = AsyncOpenAI(base_url=self.api_base, api_key=self.api_key)
        else:
            self.client = AsyncOpenAI(api_key=self.api_key)
        
        logger.info(f"Initialized MetaLLM with model: {model}")
    
    async def generate_initial_seed_prompt(
        self,
        problem_description: str,
        initial_code: str,
        evaluation_criteria: str,
    ) -> str:
        """
        Generate the initial seed prompt for evolution
        
        Args:
            problem_description: Description of the problem to solve
            initial_code: The starting code (truncated if too long)
            evaluation_criteria: How solutions are evaluated
            
        Returns:
            Generated seed prompt
        """
        # Truncate initial code if too long
        if len(initial_code) > 3000:
            initial_code = initial_code[:3000] + "\n... (truncated)"
        
        prompt = INITIAL_SEED_PROMPT_TEMPLATE.format(
            problem_description=problem_description,
            initial_code=initial_code,
            evaluation_criteria=evaluation_criteria,
        )
        
        logger.info("Generating initial seed prompt...")
        
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                max_tokens=2000,
            )
            
            seed_prompt = response.choices[0].message.content.strip()
            logger.info(f"Generated initial seed prompt ({len(seed_prompt)} chars)")
            
            return seed_prompt
            
        except Exception as e:
            logger.error(f"Failed to generate initial seed prompt: {e}")
            # Return a basic fallback prompt
            return self._get_fallback_prompt(problem_description)
    
    async def refine_seed_prompt(
        self,
        seed_prompt_history: SeedPromptHistory,
        problem_description: str,
        current_best_prompt: str,
        best_program_code: str = "",
    ) -> str:
        """
        Refine the seed prompt based on convergence feedback
        
        Args:
            seed_prompt_history: History of previous prompts and results
            problem_description: Description of the problem
            current_best_prompt: The best performing prompt so far
            best_program_code: The best program code found so far
            
        Returns:
            Refined seed prompt
        """
        # Gather analysis from history
        successful_patterns = seed_prompt_history.get_successful_patterns()
        stuck_patterns = seed_prompt_history.get_stuck_patterns()
        
        # Get ALL failure modes across all entries
        all_failure_modes = []
        for entry in seed_prompt_history.entries:
            if entry.convergence_metrics.failure_modes:
                all_failure_modes.extend(entry.convergence_metrics.failure_modes)
            if entry.convergence_metrics.stuck_patterns:
                all_failure_modes.extend(entry.convergence_metrics.stuck_patterns)
        # Deduplicate and limit
        failure_modes = list(set(all_failure_modes))[:5]
        
        # Get best metrics
        best_score = 0.0
        if seed_prompt_history.best_prompt_index >= 0:
            best_entry = seed_prompt_history.entries[seed_prompt_history.best_prompt_index]
            best_score = best_entry.convergence_metrics.final_best_score
        
        # Generate code summary (not full code)
        best_code_summary = self._summarize_code(best_program_code)
        
        # Format stuck patterns concisely
        stuck_summary = "None observed yet"
        if stuck_patterns or failure_modes:
            all_stuck = stuck_patterns + failure_modes
            stuck_summary = "\n".join(f"- DO NOT: {p[:100]}" for p in all_stuck[:5])
        
        # Format successful patterns
        success_summary = "None identified yet"
        if successful_patterns:
            success_summary = "\n".join(f"- {p}" for p in successful_patterns[:3])
        
        prompt = REFINE_SEED_PROMPT_TEMPLATE.format(
            problem_description=problem_description[:500],
            current_best_prompt=current_best_prompt[:1500],  # Truncate to save context
            best_final_score=f"{best_score:.4f}",
            best_code_summary=best_code_summary,
            successful_patterns=success_summary,
            stuck_patterns=stuck_summary,
        )
        
        logger.info("Refining seed prompt based on convergence feedback...")
        
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                max_tokens=2000,
            )
            
            refined_prompt = response.choices[0].message.content.strip()
            logger.info(f"Generated refined seed prompt ({len(refined_prompt)} chars)")
            
            return refined_prompt
            
        except Exception as e:
            logger.error(f"Failed to refine seed prompt: {e}")
            # Return the current best with minor modification
            return current_best_prompt + "\n\nNote: Focus on producing valid solutions that satisfy all constraints."
    
    def _summarize_code(self, code: str) -> str:
        """
        Generate a concise summary of code: key functions and approach
        
        Args:
            code: The full program code
            
        Returns:
            A brief summary (under 500 chars)
        """
        if not code:
            return "(No valid program found yet)"
        
        lines = code.split('\n')
        
        # Extract function/class definitions
        definitions = []
        for line in lines:
            stripped = line.strip()
            if stripped.startswith('def '):
                # Get function name and signature
                func_sig = stripped[4:].split(':')[0]
                definitions.append(f"def {func_sig}")
            elif stripped.startswith('class '):
                class_name = stripped[6:].split(':')[0].split('(')[0]
                definitions.append(f"class {class_name}")
        
        # Extract key imports (for approach detection)
        imports = []
        for line in lines[:30]:  # Check first 30 lines
            if line.startswith('import ') or line.startswith('from '):
                imports.append(line.strip())
        
        # Build summary
        summary_parts = []
        
        if imports:
            key_imports = [i for i in imports if any(k in i for k in ['numpy', 'scipy', 'math', 'random', 'itertools'])]
            if key_imports:
                summary_parts.append(f"Uses: {', '.join(key_imports[:3])}")
        
        if definitions:
            summary_parts.append(f"Functions: {', '.join(definitions[:5])}")
        
        # Add first few lines of actual code (skip imports/comments)
        code_preview = []
        for line in lines:
            stripped = line.strip()
            if stripped and not stripped.startswith('#') and not stripped.startswith('import') and not stripped.startswith('from'):
                code_preview.append(line)
                if len(code_preview) >= 5:
                    break
        
        if code_preview:
            summary_parts.append(f"Approach preview:\n```\n{chr(10).join(code_preview)}\n```")
        
        return "\n".join(summary_parts) if summary_parts else "(Unable to summarize code)"
    
    def _get_fallback_prompt(self, problem_description: str) -> str:
        """Generate a basic fallback prompt if LLM fails"""
        return f"""You are an expert programmer tasked with improving code solutions.

## Goal
{problem_description}

## Guidelines
1. Make targeted improvements that increase the performance metrics
2. Ensure all constraints are satisfied (validity is critical)
3. Try different algorithmic approaches if current approach is stuck
4. Learn from previous attempts - avoid repeating the same mistakes
5. Balance exploration of new ideas with exploitation of working patterns

Focus on correctness first, then optimize for performance."""

