"""
Advanced Data Quality Framework for Retail Data Warehouse
This module provides comprehensive data quality checks, validation rules,
and monitoring capabilities for the ETL pipeline.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any
import logging
from dataclasses import dataclass
from enum import Enum
import json

# Setup logging
logger = logging.getLogger(__name__)

class QualityCheckType(Enum):
    """Types of data quality checks"""
    COMPLETENESS = "completeness"
    ACCURACY = "accuracy"
    CONSISTENCY = "consistency"
    VALIDITY = "validity"
    UNIQUENESS = "uniqueness"
    TIMELINESS = "timeliness"

class SeverityLevel(Enum):
    """Severity levels for quality issues"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"

@dataclass
class QualityCheckResult:
    """Result of a data quality check"""
    check_name: str
    table_name: str
    check_type: QualityCheckType
    severity: SeverityLevel
    records_checked: int
    records_failed: int
    failure_rate: float
    error_message: str
    check_timestamp: datetime
    etl_run_id: str
    source_system: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert QualityCheckResult to dictionary for JSON serialization"""
        return {
            'check_name': self.check_name,
            'table_name': self.table_name,
            'check_type': self.check_type.value,
            'severity': self.severity.value,
            'records_checked': int(self.records_checked),
            'records_failed': int(self.records_failed),
            'failure_rate': float(self.failure_rate),
            'error_message': self.error_message,
            'check_timestamp': self.check_timestamp.isoformat(),
            'etl_run_id': self.etl_run_id,
            'source_system': self.source_system
        }

class DataQualityFramework:
    """Comprehensive Data Quality Framework"""
    
    def __init__(self, etl_run_id: str = None):
        self.etl_run_id = etl_run_id or f"ETL_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.quality_results: List[QualityCheckResult] = []
        self.quality_rules = self._load_quality_rules()
        
    def _load_quality_rules(self) -> Dict:
        """Load data quality rules configuration"""
        return {
            'completeness': {
                'required_columns': {
                    'FactSales': ['DateKey', 'CustomerKey', 'ProductKey', 'LocationKey', 'Quantity', 'UnitPrice', 'TotalRevenue'],
                    'DimCustomer': ['CustomerKey', 'CustomerID', 'Country'],
                    'DimProduct': ['ProductKey', 'StockCode'],
                    'DimDate': ['DateKey', 'FullDate', 'Year', 'Month'],
                    'DimLocation': ['LocationKey', 'Country']
                },
                'threshold': 0.95  # 95% completeness required
            },
            'accuracy': {
                'numeric_ranges': {
                    'Quantity': {'min': 1, 'max': 100000},
                    'UnitPrice': {'min': 0.01, 'max': 50000.0},
                    'TotalRevenue': {'min': 0.01, 'max': 1000000.0}
                },
                'date_ranges': {
                    'InvoiceDate': {'min': '2010-01-01', 'max': '2012-12-31'}
                }
            },
            'consistency': {
                'business_rules': [
                    'TotalRevenue = Quantity * UnitPrice',
                    'CustomerKey must exist in DimCustomer',
                    'ProductKey must exist in DimProduct',
                    'DateKey must exist in DimDate'
                ]
            },
            'validity': {
                'format_checks': {
                    'InvoiceNo': r'^[0-9]+$|^C[0-9]+$',  # Numeric or starts with C
                    'StockCode': r'^[A-Z0-9]+$',  # Alphanumeric uppercase
                    'CustomerID': r'^[0-9]+$|^UNKNOWN$'  # Numeric or UNKNOWN
                }
            }
        }
    
    def run_completeness_checks(self, df: pd.DataFrame, table_name: str) -> List[QualityCheckResult]:
        """Run completeness checks on a DataFrame"""
        results = []
        rules = self.quality_rules['completeness']
        
        if table_name in rules['required_columns']:
            required_cols = rules['required_columns'][table_name]
            threshold = rules['threshold']
            
            for col in required_cols:
                if col in df.columns:
                    total_records = len(df)
                    null_records = df[col].isnull().sum()
                    completeness_rate = (total_records - null_records) / total_records
                    
                    severity = SeverityLevel.CRITICAL if completeness_rate < threshold else SeverityLevel.LOW
                    
                    result = QualityCheckResult(
                        check_name=f"Completeness_{col}",
                        table_name=table_name,
                        check_type=QualityCheckType.COMPLETENESS,
                        severity=severity,
                        records_checked=total_records,
                        records_failed=null_records,
                        failure_rate=1 - completeness_rate,
                        error_message=f"Column {col} has {null_records} null values ({completeness_rate:.2%} completeness)",
                        check_timestamp=datetime.now(),
                        etl_run_id=self.etl_run_id,
                        source_system="ETL_Pipeline"
                    )
                    results.append(result)
                    self.quality_results.append(result)
        
        return results
    
    def run_accuracy_checks(self, df: pd.DataFrame, table_name: str) -> List[QualityCheckResult]:
        """Run accuracy checks on a DataFrame"""
        results = []
        rules = self.quality_rules['accuracy']
        
        # Numeric range checks
        if 'numeric_ranges' in rules:
            for col, range_config in rules['numeric_ranges'].items():
                if col in df.columns:
                    total_records = len(df)
                    invalid_records = 0
                    
                    # Check minimum value
                    if 'min' in range_config:
                        invalid_records += (df[col] < range_config['min']).sum()
                    
                    # Check maximum value
                    if 'max' in range_config:
                        invalid_records += (df[col] > range_config['max']).sum()
                    
                    failure_rate = invalid_records / total_records if total_records > 0 else 0
                    
                    severity = SeverityLevel.HIGH if failure_rate > 0.05 else SeverityLevel.MEDIUM
                    
                    result = QualityCheckResult(
                        check_name=f"Accuracy_{col}_Range",
                        table_name=table_name,
                        check_type=QualityCheckType.ACCURACY,
                        severity=severity,
                        records_checked=total_records,
                        records_failed=invalid_records,
                        failure_rate=failure_rate,
                        error_message=f"Column {col} has {invalid_records} values outside range [{range_config.get('min', 'N/A')}, {range_config.get('max', 'N/A')}]",
                        check_timestamp=datetime.now(),
                        etl_run_id=self.etl_run_id,
                        source_system="ETL_Pipeline"
                    )
                    results.append(result)
                    self.quality_results.append(result)
        
        return results
    
    def run_consistency_checks(self, df: pd.DataFrame, table_name: str) -> List[QualityCheckResult]:
        """Run consistency checks on a DataFrame"""
        results = []
        
        if table_name == 'FactSales':
            # Check TotalRevenue = Quantity * UnitPrice
            if all(col in df.columns for col in ['Quantity', 'UnitPrice', 'TotalRevenue']):
                calculated_revenue = df['Quantity'] * df['UnitPrice']
                inconsistent_records = abs(df['TotalRevenue'] - calculated_revenue) > 0.01
                inconsistent_count = inconsistent_records.sum()
                
                if inconsistent_count > 0:
                    result = QualityCheckResult(
                        check_name="Consistency_Revenue_Calculation",
                        table_name=table_name,
                        check_type=QualityCheckType.CONSISTENCY,
                        severity=SeverityLevel.CRITICAL,
                        records_checked=len(df),
                        records_failed=inconsistent_count,
                        failure_rate=inconsistent_count / len(df),
                        error_message=f"TotalRevenue calculation inconsistent in {inconsistent_count} records",
                        check_timestamp=datetime.now(),
                        etl_run_id=self.etl_run_id,
                        source_system="ETL_Pipeline"
                    )
                    results.append(result)
                    self.quality_results.append(result)
        
        return results
    
    def run_validity_checks(self, df: pd.DataFrame, table_name: str) -> List[QualityCheckResult]:
        """Run validity checks on a DataFrame"""
        results = []
        rules = self.quality_rules['validity']
        
        if 'format_checks' in rules:
            for col, pattern in rules['format_checks'].items():
                if col in df.columns:
                    total_records = len(df)
                    invalid_records = ~df[col].astype(str).str.match(pattern, na=False)
                    invalid_count = invalid_records.sum()
                    
                    if invalid_count > 0:
                        severity = SeverityLevel.MEDIUM if invalid_count / total_records < 0.1 else SeverityLevel.HIGH
                        
                        result = QualityCheckResult(
                            check_name=f"Validity_{col}_Format",
                            table_name=table_name,
                            check_type=QualityCheckType.VALIDITY,
                            severity=severity,
                            records_checked=total_records,
                            records_failed=invalid_count,
                            failure_rate=invalid_count / total_records,
                            error_message=f"Column {col} has {invalid_count} records with invalid format",
                            check_timestamp=datetime.now(),
                            etl_run_id=self.etl_run_id,
                            source_system="ETL_Pipeline"
                        )
                        results.append(result)
                        self.quality_results.append(result)
        
        return results
    
    def run_uniqueness_checks(self, df: pd.DataFrame, table_name: str, key_columns: List[str]) -> List[QualityCheckResult]:
        """Run uniqueness checks on key columns"""
        results = []
        
        for col in key_columns:
            if col in df.columns:
                total_records = len(df)
                unique_records = df[col].nunique()
                duplicate_records = total_records - unique_records
                
                if duplicate_records > 0:
                    result = QualityCheckResult(
                        check_name=f"Uniqueness_{col}",
                        table_name=table_name,
                        check_type=QualityCheckType.UNIQUENESS,
                        severity=SeverityLevel.HIGH,
                        records_checked=total_records,
                        records_failed=duplicate_records,
                        failure_rate=duplicate_records / total_records,
                        error_message=f"Column {col} has {duplicate_records} duplicate values",
                        check_timestamp=datetime.now(),
                        etl_run_id=self.etl_run_id,
                        source_system="ETL_Pipeline"
                    )
                    results.append(result)
                    self.quality_results.append(result)
        
        return results
    
    def run_referential_integrity_checks(self, fact_df: pd.DataFrame, dim_tables: Dict[str, pd.DataFrame]) -> List[QualityCheckResult]:
        """Run referential integrity checks between fact and dimension tables"""
        results = []
        
        # Check foreign key relationships
        foreign_key_mappings = {
            'DateKey': 'DimDate',
            'CustomerKey': 'DimCustomer',
            'ProductKey': 'DimProduct',
            'LocationKey': 'DimLocation'
        }
        
        for fk_col, dim_table in foreign_key_mappings.items():
            if fk_col in fact_df.columns and dim_table in dim_tables:
                fact_values = set(fact_df[fk_col].dropna().unique())
                dim_values = set(dim_tables[dim_table]['DateKey' if fk_col == 'DateKey' else fk_col].unique())
                
                orphaned_values = fact_values - dim_values
                orphaned_count = len(orphaned_values)
                
                if orphaned_count > 0:
                    result = QualityCheckResult(
                        check_name=f"Referential_Integrity_{fk_col}",
                        table_name="FactSales",
                        check_type=QualityCheckType.CONSISTENCY,
                        severity=SeverityLevel.CRITICAL,
                        records_checked=len(fact_df),
                        records_failed=orphaned_count,
                        failure_rate=orphaned_count / len(fact_df),
                        error_message=f"Found {orphaned_count} orphaned {fk_col} values in FactSales",
                        check_timestamp=datetime.now(),
                        etl_run_id=self.etl_run_id,
                        source_system="ETL_Pipeline"
                    )
                    results.append(result)
                    self.quality_results.append(result)
        
        return results
    
    def run_all_checks(self, df: pd.DataFrame, table_name: str, key_columns: List[str] = None) -> List[QualityCheckResult]:
        """Run all data quality checks on a DataFrame"""
        logger.info(f"Running data quality checks for table: {table_name}")
        
        all_results = []
        
        # Run all check types
        all_results.extend(self.run_completeness_checks(df, table_name))
        all_results.extend(self.run_accuracy_checks(df, table_name))
        all_results.extend(self.run_consistency_checks(df, table_name))
        all_results.extend(self.run_validity_checks(df, table_name))
        
        if key_columns:
            all_results.extend(self.run_uniqueness_checks(df, table_name, key_columns))
        
        return all_results
    
    def get_quality_summary(self) -> Dict[str, Any]:
        """Get a summary of all quality check results"""
        if not self.quality_results:
            return {"message": "No quality checks have been run yet"}
        
        total_checks = len(self.quality_results)
        failed_checks = len([r for r in self.quality_results if r.records_failed > 0])
        
        severity_counts = {
            'critical': len([r for r in self.quality_results if r.severity == SeverityLevel.CRITICAL]),
            'high': len([r for r in self.quality_results if r.severity == SeverityLevel.HIGH]),
            'medium': len([r for r in self.quality_results if r.severity == SeverityLevel.MEDIUM]),
            'low': len([r for r in self.quality_results if r.severity == SeverityLevel.LOW])
        }
        
        check_type_counts = {}
        for check_type in QualityCheckType:
            check_type_counts[check_type.value] = len([r for r in self.quality_results if r.check_type == check_type])
        
        return {
            'etl_run_id': self.etl_run_id,
            'total_checks': int(total_checks),
            'failed_checks': int(failed_checks),
            'success_rate': float((total_checks - failed_checks) / total_checks if total_checks > 0 else 0),
            'severity_breakdown': {k: int(v) for k, v in severity_counts.items()},
            'check_type_breakdown': {k: int(v) for k, v in check_type_counts.items()},
            'critical_issues': [r.to_dict() for r in self.quality_results if r.severity == SeverityLevel.CRITICAL],
            'high_issues': [r.to_dict() for r in self.quality_results if r.severity == SeverityLevel.HIGH]
        }
    
    def export_quality_report(self, file_path: str = None) -> str:
        """Export quality check results to JSON file"""
        if file_path is None:
            file_path = f"data_quality_report_{self.etl_run_id}.json"
        
        report_data = {
            'summary': self.get_quality_summary(),
            'detailed_results': [
                {
                    'check_name': r.check_name,
                    'table_name': r.table_name,
                    'check_type': r.check_type.value,
                    'severity': r.severity.value,
                    'records_checked': r.records_checked,
                    'records_failed': r.records_failed,
                    'failure_rate': r.failure_rate,
                    'error_message': r.error_message,
                    'check_timestamp': r.check_timestamp.isoformat(),
                    'etl_run_id': r.etl_run_id,
                    'source_system': r.source_system
                }
                for r in self.quality_results
            ]
        }
        
        with open(file_path, 'w') as f:
            json.dump(report_data, f, indent=2, default=str)
        
        logger.info(f"Quality report exported to: {file_path}")
        return file_path
    
    def should_proceed_with_load(self, critical_threshold: float = 0.05) -> bool:
        """Determine if ETL should proceed based on quality check results"""
        critical_issues = [r for r in self.quality_results if r.severity == SeverityLevel.CRITICAL]
        
        if not critical_issues:
            return True
        
        # Calculate overall failure rate for critical issues
        total_critical_records = sum(r.records_checked for r in critical_issues)
        total_critical_failures = sum(r.records_failed for r in critical_issues)
        
        if total_critical_records > 0:
            critical_failure_rate = total_critical_failures / total_critical_records
            return critical_failure_rate <= critical_threshold
        
        return True


# ==================== DATA PROFILING ====================

class DataProfiler:
    """Advanced data profiling capabilities"""
    
    @staticmethod
    def profile_dataframe(df: pd.DataFrame, table_name: str) -> Dict[str, Any]:
        """Generate comprehensive data profile for a DataFrame"""
        profile = {
            'table_name': table_name,
            'total_records': int(len(df)),
            'total_columns': int(len(df.columns)),
            'memory_usage': int(df.memory_usage(deep=True).sum()),
            'columns': {}
        }
        
        for col in df.columns:
            col_profile = {
                'data_type': str(df[col].dtype),
                'null_count': int(df[col].isnull().sum()),
                'null_percentage': float((df[col].isnull().sum() / len(df)) * 100),
                'unique_count': int(df[col].nunique()),
                'unique_percentage': float((df[col].nunique() / len(df)) * 100)
            }
            
            # Numeric columns
            if df[col].dtype in ['int64', 'float64']:
                col_profile.update({
                    'min_value': float(df[col].min()) if pd.notna(df[col].min()) else None,
                    'max_value': float(df[col].max()) if pd.notna(df[col].max()) else None,
                    'mean_value': float(df[col].mean()) if pd.notna(df[col].mean()) else None,
                    'median_value': float(df[col].median()) if pd.notna(df[col].median()) else None,
                    'std_value': float(df[col].std()) if pd.notna(df[col].std()) else None,
                    'zero_count': int((df[col] == 0).sum()),
                    'negative_count': int((df[col] < 0).sum())
                })
            
            # String columns
            elif df[col].dtype == 'object':
                col_profile.update({
                    'min_length': int(df[col].astype(str).str.len().min()) if pd.notna(df[col].astype(str).str.len().min()) else None,
                    'max_length': int(df[col].astype(str).str.len().max()) if pd.notna(df[col].astype(str).str.len().max()) else None,
                    'avg_length': float(df[col].astype(str).str.len().mean()) if pd.notna(df[col].astype(str).str.len().mean()) else None,
                    'empty_string_count': int((df[col].astype(str) == '').sum()),
                    'whitespace_count': int(df[col].astype(str).str.strip().eq('').sum())
                })
            
            # Date columns
            elif 'datetime' in str(df[col].dtype):
                col_profile.update({
                    'min_date': str(df[col].min()) if pd.notna(df[col].min()) else None,
                    'max_date': str(df[col].max()) if pd.notna(df[col].max()) else None,
                    'date_range_days': int((df[col].max() - df[col].min()).days) if pd.notna(df[col].max()) and pd.notna(df[col].min()) else None
                })
            
            profile['columns'][col] = col_profile
        
        return profile
    
    @staticmethod
    def detect_outliers(df: pd.DataFrame, numeric_columns: List[str]) -> Dict[str, List[int]]:
        """Detect outliers in numeric columns using IQR method"""
        outliers = {}
        
        for col in numeric_columns:
            if col in df.columns and df[col].dtype in ['int64', 'float64']:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outlier_indices = df[(df[col] < lower_bound) | (df[col] > upper_bound)].index.tolist()
                outliers[col] = outlier_indices
        
        return outliers


if __name__ == '__main__':
    # Example usage
    dq_framework = DataQualityFramework()
    
    # Create sample data for testing
    sample_data = pd.DataFrame({
        'id': [1, 2, 3, 4, 5],
        'name': ['Alice', 'Bob', None, 'David', 'Eve'],
        'age': [25, 30, 35, 40, 45],
        'salary': [50000, 60000, 70000, 80000, 90000]
    })
    
    # Run quality checks
    results = dq_framework.run_all_checks(sample_data, 'SampleTable', ['id'])
    
    # Print summary
    summary = dq_framework.get_quality_summary()
    print(json.dumps(summary, indent=2, default=str))
