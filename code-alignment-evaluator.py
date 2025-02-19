import ast
import difflib
import os
import subprocess
import sys
import json
from typing import Dict, List, Tuple, Any, Optional
import re
import astor  # for AST to source conversion
import astunparse  # alternative AST to source converter

class CodeAlignmentEvaluator:
    """
    A comprehensive evaluator to measure alignment between LLM-generated code and gold standard code.
    """
    
    def __init__(self, generated_file: str, gold_file: str, test_file: Optional[str] = None):
        """
        Initialize the evaluator with file paths.
        
        Args:
            generated_file: Path to the LLM-generated code file
            gold_file: Path to the gold standard code file
            test_file: Optional path to test file for functional testing
        """
        self.generated_file = generated_file
        self.gold_file = gold_file
        self.test_file = test_file
        
        # Read file contents
        with open(generated_file, 'r') as f:
            self.generated_code = f.read()
        
        with open(gold_file, 'r') as f:
            self.gold_code = f.read()
            
        # Parse ASTs
        try:
            self.generated_ast = ast.parse(self.generated_code)
            self.gold_ast = ast.parse(self.gold_code)
            self.ast_parse_error = None
        except SyntaxError as e:
            self.ast_parse_error = str(e)
            
    def evaluate(self) -> Dict[str, Any]:
        """
        Run all evaluations and return comprehensive results.
        
        Returns:
            Dictionary containing all evaluation metrics
        """
        results = {
            "syntax_similarity": self.evaluate_syntax(),
            "structural_similarity": self.evaluate_structure(),
            "semantic_similarity": self.evaluate_semantics(),
            "variable_naming": self.evaluate_variable_naming(),
            "code_complexity": self.evaluate_complexity(),
            "functional_equivalence": self.evaluate_functional_equivalence(),
            "overall_score": 0.0  # Will be calculated at the end
        }
        
        # Calculate weighted overall score
        weights = {
            "syntax_similarity": 0.1,
            "structural_similarity": 0.2,
            "semantic_similarity": 0.3,
            "variable_naming": 0.1,
            "code_complexity": 0.1,
            "functional_equivalence": 0.2
        }
        
        weighted_sum = 0
        valid_metrics = 0
        
        for metric, weight in weights.items():
            if results[metric] is not None:
                weighted_sum += results[metric] * weight
                valid_metrics += weight
                
        if valid_metrics > 0:
            results["overall_score"] = weighted_sum / valid_metrics
            
        return results
    
    def evaluate_syntax(self) -> float:
        """
        Evaluate basic syntax correctness and similarity.
        
        Returns:
            Float score between 0.0 and 1.0, or None if parsing failed
        """
        if self.ast_parse_error:
            return 0.0
            
        # Use difflib to get a basic text similarity score
        matcher = difflib.SequenceMatcher(None, self.generated_code, self.gold_code)
        return matcher.ratio()
    
    def evaluate_structure(self) -> float:
        """
        Evaluate structural similarity by comparing AST structures.
        
        Returns:
            Float score between 0.0 and 1.0, or None if parsing failed
        """
        if self.ast_parse_error:
            return 0.0
            
        # Count node types in both ASTs and compare distributions
        def count_node_types(root):
            counter = {}
            for node in ast.walk(root):
                node_type = type(node).__name__
                counter[node_type] = counter.get(node_type, 0) + 1
            return counter
            
        gen_counts = count_node_types(self.generated_ast)
        gold_counts = count_node_types(self.gold_ast)
        
        # Calculate Jaccard similarity between node type distributions
        all_types = set(list(gen_counts.keys()) + list(gold_counts.keys()))
        intersection = 0
        union = 0
        
        for node_type in all_types:
            gen_count = gen_counts.get(node_type, 0)
            gold_count = gold_counts.get(node_type, 0)
            intersection += min(gen_count, gold_count)
            union += max(gen_count, gold_count)
            
        return intersection / union if union > 0 else 0.0
    
    def evaluate_semantics(self) -> float:
        """
        Evaluate semantic similarity by comparing normalized code.
        
        Returns:
            Float score between 0.0 and 1.0, or None if parsing failed
        """
        if self.ast_parse_error:
            return 0.0
            
        # Normalize code by:
        # 1. Converting ASTs back to source with consistent formatting
        # 2. Removing comments and normalizing whitespace
        # 3. Normalizing variable names
        
        try:
            # Convert ASTs back to source code with consistent formatting
            normalized_generated = astor.to_source(self.generated_ast)
            normalized_gold = astor.to_source(self.gold_ast)
            
            # Remove comments and normalize whitespace
            def normalize_code(code_str):
                # Remove comments
                code_str = re.sub(r'#.*$', '', code_str, flags=re.MULTILINE)
                # Normalize whitespace
                code_str = re.sub(r'\s+', ' ', code_str).strip()
                return code_str
                
            normalized_generated = normalize_code(normalized_generated)
            normalized_gold = normalize_code(normalized_gold)
            
            # Compute similarity on normalized code
            matcher = difflib.SequenceMatcher(None, normalized_generated, normalized_gold)
            return matcher.ratio()
            
        except Exception:
            # Fallback if astor fails
            try:
                normalized_generated = astunparse.unparse(self.generated_ast)
                normalized_gold = astunparse.unparse(self.gold_ast)
                
                normalized_generated = normalize_code(normalized_generated)
                normalized_gold = normalize_code(normalized_gold)
                
                matcher = difflib.SequenceMatcher(None, normalized_generated, normalized_gold)
                return matcher.ratio()
            except Exception:
                return 0.5  # Default middle value if normalization fails
    
    def evaluate_variable_naming(self) -> float:
        """
        Evaluate similarity in variable naming conventions.
        
        Returns:
            Float score between 0.0 and 1.0, or None if parsing failed
        """
        if self.ast_parse_error:
            return 0.0
            
        # Extract variable names from both ASTs
        def extract_variable_names(root):
            variables = set()
            
            for node in ast.walk(root):
                # Variable assignments
                if isinstance(node, ast.Assign):
                    for target in node.targets:
                        if isinstance(target, ast.Name):
                            variables.add(target.id)
                # Function parameters
                elif isinstance(node, ast.FunctionDef):
                    for arg in node.args.args:
                        variables.add(arg.arg)
                
            return variables
            
        gen_vars = extract_variable_names(self.generated_ast)
        gold_vars = extract_variable_names(self.gold_ast)
        
        # Calculate naming convention similarity
        def get_naming_convention(var_name):
            if var_name.isupper():
                return "UPPER_CASE"
            elif var_name.islower():
                return "lower_case" 
            elif var_name[0].islower() and "_" in var_name:
                return "snake_case"
            elif var_name[0].islower() and any(c.isupper() for c in var_name):
                return "camelCase"
            elif var_name[0].isupper() and any(c.isupper() for c in var_name[1:]):
                return "PascalCase"
            else:
                return "other"
                
        gen_conventions = [get_naming_convention(var) for var in gen_vars]
        gold_conventions = [get_naming_convention(var) for var in gold_vars]
        
        gen_convention_counts = {conv: gen_conventions.count(conv) for conv in set(gen_conventions)}
        gold_convention_counts = {conv: gold_conventions.count(conv) for conv in set(gold_conventions)}
        
        # Calculate convention similarity
        all_conventions = set(list(gen_convention_counts.keys()) + list(gold_convention_counts.keys()))
        if not all_conventions:
            return 1.0  # No variables in either code
            
        similarity_sum = 0
        for conv in all_conventions:
            gen_freq = gen_convention_counts.get(conv, 0) / len(gen_conventions) if gen_conventions else 0
            gold_freq = gold_convention_counts.get(conv, 0) / len(gold_conventions) if gold_conventions else 0
            similarity_sum += 1.0 - abs(gen_freq - gold_freq)
            
        return similarity_sum / len(all_conventions)
    
    def evaluate_complexity(self) -> float:
        """
        Evaluate similarity in code complexity.
        
        Returns:
            Float score between 0.0 and 1.0 representing how similar the complexity is
        """
        if self.ast_parse_error:
            return 0.0
            
        # Calculate cyclomatic complexity
        def calculate_complexity(root):
            complexity = 1  # Base complexity
            
            for node in ast.walk(root):
                # Control flow increases complexity
                if isinstance(node, (ast.If, ast.While, ast.For)):
                    complexity += 1
                # Each logical operator adds complexity
                elif isinstance(node, ast.BoolOp):
                    complexity += len(node.values) - 1
                # Each except handler adds complexity
                elif isinstance(node, ast.Try):
                    complexity += len(node.handlers)
                    
            return complexity
            
        gen_complexity = calculate_complexity(self.generated_ast)
        gold_complexity = calculate_complexity(self.gold_ast)
        
        # Calculate relative complexity similarity (inverse of normalized difference)
        max_complexity = max(gen_complexity, gold_complexity)
        if max_complexity == 0:
            return 1.0  # Both have zero complexity
            
        complexity_diff = abs(gen_complexity - gold_complexity) / max_complexity
        return 1.0 - complexity_diff
    
    def evaluate_functional_equivalence(self) -> float:
        """
        Evaluate functional equivalence by running tests if provided.
        
        Returns:
            Float score between 0.0 and 1.0, or None if tests couldn't be run
        """
        if not self.test_file or self.ast_parse_error:
            # Use static analysis as a fallback
            return self.evaluate_semantics()
            
        # Create temporary directories for testing
        import tempfile
        import shutil
        
        temp_dir_gold = tempfile.mkdtemp()
        temp_dir_gen = tempfile.mkdtemp()
        
        try:
            # Copy test file to both directories
            shutil.copy(self.test_file, os.path.join(temp_dir_gold, 'test.py'))
            shutil.copy(self.test_file, os.path.join(temp_dir_gen, 'test.py'))
            
            # Create gold and generated modules in respective directories
            with open(os.path.join(temp_dir_gold, 'solution.py'), 'w') as f:
                f.write(self.gold_code)
                
            with open(os.path.join(temp_dir_gen, 'solution.py'), 'w') as f:
                f.write(self.generated_code)
                
            # Run tests on both implementations
            gold_result = self._run_tests(temp_dir_gold)
            gen_result = self._run_tests(temp_dir_gen)
            
            if gold_result['success'] and gen_result['success']:
                # Both passed all tests - perfect score
                return 1.0
            elif not gold_result['success'] and not gen_result['success']:
                # Both failed - compare error similarity
                error_sim = difflib.SequenceMatcher(None, 
                                                   gold_result.get('error', ''),
                                                   gen_result.get('error', '')).ratio()
                return 0.5 * error_sim  # Partial score for similar errors
            elif gold_result['success'] and not gen_result['success']:
                # Gold works, generated doesn't - check partial correctness
                return 0.0
            else:
                # Generated works, gold doesn't (unusual case)
                return 0.5
                
        finally:
            # Clean up temporary directories
            shutil.rmtree(temp_dir_gold)
            shutil.rmtree(temp_dir_gen)
    
    def _run_tests(self, directory: str) -> Dict[str, Any]:
        """Helper to run tests in the specified directory."""
        result = {'success': False, 'error': None}
        
        try:
            # Run the test file
            process = subprocess.run(
                [sys.executable, 'test.py'],
                cwd=directory,
                capture_output=True,
                text=True,
                timeout=10  # Prevent infinite loops
            )
            
            if process.returncode == 0:
                result['success'] = True
            else:
                result['success'] = False
                result['error'] = process.stderr or process.stdout
                
        except subprocess.TimeoutExpired:
            result['success'] = False
            result['error'] = "Execution timed out"
        except Exception as e:
            result['success'] = False
            result['error'] = str(e)
            
        return result


def main():
    """CLI interface for the evaluator."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate alignment between generated and gold standard code')
    parser.add_argument('--generated', required=True, help='Path to the generated code file')
    parser.add_argument('--gold', required=True, help='Path to the gold standard code file')
    parser.add_argument('--test', help='Optional path to test file for functional testing')
    parser.add_argument('--output', help='Path to save results JSON (prints to stdout if not specified)')
    
    args = parser.parse_args()
    
    evaluator = CodeAlignmentEvaluator(args.generated, args.gold, args.test)
    results = evaluator.evaluate()
    
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
    else:
        print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
