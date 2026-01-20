"""
Tier 1: Automated Structural Validation for LLM outputs
Implements schema compliance, enum validation, and structural completeness metrics
"""

from pydantic import ValidationError
from typing import Dict, List, Any, Optional
import json
from data.activity_record_model import ActivityRecord
from data.choice_data_enums import (
    SustainableDevelopmentGoal,
    TargetPopulation,
    FocusAreaCategory,
    GoalOutput,
    GoalInstitutionalOutcome,
    GoalCommunityImpact,
    ActivityType as ActivityTypeEnum
)


class Tier1Evaluator:
    """Tier 1: Automated Structural Validation"""
    
    def __init__(self):
        # Define all enum fields and their valid values
        self.enum_fields = {
            'programsOrInitiatives': [e.value for e in SustainableDevelopmentGoal],
            'targetPopulation': [e.value for e in TargetPopulation],
            'focusAreaCategories': [e.value for e in FocusAreaCategory],
        }
        
        # Define goal enum values
        self.goal_enums = {
            'outputs': [e.value for e in GoalOutput],
            'institutionalOutcomes': [e.value for e in GoalInstitutionalOutcome],
            'communityImpacts': [e.value for e in GoalCommunityImpact]
        }
    
    def validate_schema_compliance(self, output_json: Dict[str, Any]) -> Dict[str, Any]:
        """
        1.1 Schema Compliance: Validates JSON structure against Pydantic schema
        """
        metrics = {
            'is_valid_json': False,
            'has_all_required_fields': False,
            'correct_data_types': False,
            'missing_fields': [],
            'type_errors': [],
            'compliance_score': 0.0
        }
        
        try:
            # Try to validate with Pydantic model
            ActivityRecord(**output_json)
            metrics['is_valid_json'] = True
            metrics['has_all_required_fields'] = True
            metrics['correct_data_types'] = True
            metrics['compliance_score'] = 1.0
        except ValidationError as e:
            # Parse validation errors
            errors = e.errors()
            total_errors = len(errors)
            
            for error in errors:
                field_path = ' -> '.join(str(loc) for loc in error['loc'])
                error_type = error['type']
                
                if 'missing' in error_type:
                    metrics['missing_fields'].append(field_path)
                elif 'type' in error_type or 'value' in error_type:
                    metrics['type_errors'].append({
                        'field': field_path,
                        'error': error['msg']
                    })
            
            # Calculate partial compliance score
            # Start with 1.0 and deduct points for each error category
            metrics['is_valid_json'] = True  # If we got here, it's valid JSON
            metrics['has_all_required_fields'] = len(metrics['missing_fields']) == 0
            metrics['correct_data_types'] = len(metrics['type_errors']) == 0
            
            # Compliance score: 1.0 - (errors / theoretical_max_errors)
            # Assume max of 50 possible errors as baseline
            metrics['compliance_score'] = max(0.0, 1.0 - (total_errors / 50))
        except json.JSONDecodeError:
            metrics['is_valid_json'] = False
            metrics['compliance_score'] = 0.0
        except Exception as e:
            metrics['type_errors'].append({'error': str(e)})
            metrics['compliance_score'] = 0.0
        
        return metrics
    
    def validate_enum_compliance(self, output_json: Dict[str, Any]) -> Dict[str, Any]:
        """
        1.2 Enum Validation: Checks all enum fields contain only valid values
        """
        metrics = {
            'enum_fields_checked': 0,
            'enum_violations': [],
            'enum_accuracy': 0.0,
            'total_enum_values': 0,
            'valid_enum_values': 0
        }
        
        # Check top-level enum fields
        for field, valid_values in self.enum_fields.items():
            if field in output_json and output_json[field]:
                metrics['enum_fields_checked'] += 1
                field_values = output_json[field]
                
                # Handle string or list values
                if isinstance(field_values, str):
                    field_values = [field_values]
                
                for value in field_values:
                    metrics['total_enum_values'] += 1
                    if value not in valid_values:
                        metrics['enum_violations'].append({
                            'field': field,
                            'invalid_value': value,
                            'sample_valid_options': valid_values[:5]  # Show first 5 valid options
                        })
                    else:
                        metrics['valid_enum_values'] += 1
        
        # Check goal enums (nested)
        if 'goals' in output_json:
            goals = output_json['goals']
            for goal_type, valid_values in self.goal_enums.items():
                if goal_type in goals:
                    metrics['enum_fields_checked'] += 1
                    goal_section = goals[goal_type]
                    
                    # Check expected and achieved
                    for sub_field in ['expected', 'achieved']:
                        if sub_field in goal_section:
                            field_values = goal_section[sub_field]
                            if isinstance(field_values, str):
                                field_values = [field_values]
                            
                            for value in field_values:
                                metrics['total_enum_values'] += 1
                                if value not in valid_values:
                                    metrics['enum_violations'].append({
                                        'field': f'goals.{goal_type}.{sub_field}',
                                        'invalid_value': value,
                                        'sample_valid_options': valid_values[:5]
                                    })
                                else:
                                    metrics['valid_enum_values'] += 1
        
        # Calculate enum accuracy
        if metrics['total_enum_values'] > 0:
            metrics['enum_accuracy'] = metrics['valid_enum_values'] / metrics['total_enum_values']
        else:
            metrics['enum_accuracy'] = 1.0  # No enums = 100% accuracy
        
        return metrics
    
    def calculate_completeness(self, output_json: Dict[str, Any]) -> Dict[str, Any]:
        """
        1.3 Structural Completeness: Measures how many fields are populated vs empty
        """
        metrics = {
            'completeness_ratio': 0.0,
            'total_fields': 0,
            'populated_fields': 0,
            'empty_critical_fields': [],
            'empty_optional_fields': []
        }
        
        # Define critical fields (should rarely be empty)
        critical_fields = [
            'activityType',
            'activityTitle',
            'activityDescription',
            'programsOrInitiatives',
            'targetPopulation',
            'focusAreaCategories'
        ]
        
        def count_fields(obj, path='', is_critical_path=False):
            """Recursively count all fields"""
            if isinstance(obj, dict):
                for key, value in obj.items():
                    current_path = f"{path}.{key}" if path else key
                    is_critical = key in critical_fields or is_critical_path
                    count_fields(value, current_path, is_critical)
            elif isinstance(obj, list):
                if len(obj) > 0:
                    metrics['populated_fields'] += 1
                    metrics['total_fields'] += 1
                else:
                    metrics['total_fields'] += 1
                    if is_critical_path:
                        metrics['empty_critical_fields'].append(path)
                    else:
                        metrics['empty_optional_fields'].append(path)
            else:
                metrics['total_fields'] += 1
                # Check if field is populated
                if obj is not None and obj != '' and obj != []:
                    metrics['populated_fields'] += 1
                else:
                    if is_critical_path:
                        metrics['empty_critical_fields'].append(path)
                    else:
                        metrics['empty_optional_fields'].append(path)
        
        count_fields(output_json)
        
        # Calculate completeness ratio
        if metrics['total_fields'] > 0:
            metrics['completeness_ratio'] = metrics['populated_fields'] / metrics['total_fields']
        
        return metrics
    
    def evaluate(self, output_json: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run complete Tier 1 evaluation
        Returns comprehensive metrics dictionary
        """
        results = {
            'tier': 'Tier 1: Automated Structural Validation',
            'schema_compliance': self.validate_schema_compliance(output_json),
            'enum_validation': self.validate_enum_compliance(output_json),
            'structural_completeness': self.calculate_completeness(output_json)
        }
        
        # Calculate overall Tier 1 score
        schema_score = results['schema_compliance']['compliance_score']
        enum_score = results['enum_validation']['enum_accuracy']
        completeness_score = results['structural_completeness']['completeness_ratio']
        
        overall_score = (schema_score * 0.4 + enum_score * 0.4 + completeness_score * 0.2)
        overall_score = max(0.0, min(1.0, overall_score))  # Clamp between 0 and 1
        
        results['overall_tier1_score'] = overall_score
        results['overall_grade'] = self._assign_grade(overall_score)
        
        return results
    
    def _assign_grade(self, score: float) -> str:
        """Assign letter grade based on score"""
        if score >= 0.95:
            return 'A+'
        elif score >= 0.90:
            return 'A'
        elif score >= 0.85:
            return 'A-'
        elif score >= 0.80:
            return 'B+'
        elif score >= 0.75:
            return 'B'
        elif score >= 0.70:
            return 'B-'
        elif score >= 0.65:
            return 'C+'
        elif score >= 0.60:
            return 'C'
        else:
            return 'F'
    
    def print_evaluation_report(self, results: Dict[str, Any]) -> str:
        """
        Format evaluation results as a readable report
        Returns formatted string for printing
        """
        lines = []
        lines.append("\n" + "=" * 80)
        lines.append("📊 TIER 1 EVALUATION REPORT: AUTOMATED STRUCTURAL VALIDATION")
        lines.append("=" * 80)
        
        # Overall score
        lines.append(f"\n🎯 OVERALL TIER 1 SCORE: {results['overall_tier1_score']:.2%} ({results['overall_grade']})")
        
        # Schema Compliance
        lines.append("\n" + "-" * 80)
        lines.append("1️⃣  SCHEMA COMPLIANCE")
        lines.append("-" * 80)
        schema = results['schema_compliance']
        lines.append(f"   ✓ Valid JSON: {schema['is_valid_json']}")
        lines.append(f"   ✓ All Required Fields: {schema['has_all_required_fields']}")
        lines.append(f"   ✓ Correct Data Types: {schema['correct_data_types']}")
        lines.append(f"   📈 Compliance Score: {schema['compliance_score']:.2%}")
        
        if schema['missing_fields']:
            lines.append(f"   ⚠️  Missing Fields: {', '.join(schema['missing_fields'])}")
        if schema['type_errors']:
            lines.append(f"   ⚠️  Type Errors: {len(schema['type_errors'])} found")
            for error in schema['type_errors'][:3]:  # Show first 3
                lines.append(f"      - {error.get('field', 'unknown')}: {error.get('error', '')}")
        
        # Enum Validation
        lines.append("\n" + "-" * 80)
        lines.append("2️⃣  ENUM VALIDATION")
        lines.append("-" * 80)
        enum_val = results['enum_validation']
        lines.append(f"   ✓ Enum Fields Checked: {enum_val['enum_fields_checked']}")
        lines.append(f"   ✓ Total Enum Values: {enum_val['total_enum_values']}")
        lines.append(f"   ✓ Valid Enum Values: {enum_val['valid_enum_values']}")
        lines.append(f"   📈 Enum Accuracy: {enum_val['enum_accuracy']:.2%}")
        
        if enum_val['enum_violations']:
            lines.append(f"   ⚠️  Enum Violations: {len(enum_val['enum_violations'])} found")
            for violation in enum_val['enum_violations'][:3]:  # Show first 3
                lines.append(f"      - Field: {violation['field']}")
                lines.append(f"        Invalid: '{violation['invalid_value']}'")
        
        # Structural Completeness
        lines.append("\n" + "-" * 80)
        lines.append("3️⃣  STRUCTURAL COMPLETENESS")
        lines.append("-" * 80)
        completeness = results['structural_completeness']
        lines.append(f"   ✓ Total Fields: {completeness['total_fields']}")
        lines.append(f"   ✓ Populated Fields: {completeness['populated_fields']}")
        lines.append(f"   📈 Completeness Ratio: {completeness['completeness_ratio']:.2%}")
        
        if completeness['empty_critical_fields']:
            lines.append(f"   ⚠️  Empty Critical Fields: {len(completeness['empty_critical_fields'])}")
            for field in completeness['empty_critical_fields'][:5]:
                lines.append(f"      - {field}")
        
        lines.append("\n" + "=" * 80)
        lines.append("END OF TIER 1 EVALUATION REPORT")
        lines.append("=" * 80 + "\n")
        
        return '\n'.join(lines)


def evaluate_llm_output(output_json: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convenience function to evaluate LLM output with Tier 1 metrics
    """
    evaluator = Tier1Evaluator()
    return evaluator.evaluate(output_json)


def print_evaluation(output_json: Dict[str, Any]) -> str:
    """
    Convenience function to get formatted evaluation report
    """
    evaluator = Tier1Evaluator()
    results = evaluator.evaluate(output_json)
    return evaluator.print_evaluation_report(results)
