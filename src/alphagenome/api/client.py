"""
AlphaGenome Client SDK

Python client for interacting with AlphaGenome API:
- Simplified interface for genomic track prediction
- Variant scoring with batch support
- Asynchronous job management
- Error handling and retry logic
"""

import requests
import time
import json
from typing import Dict, List, Optional, Any, Union, Tuple
import pandas as pd
from pathlib import Path
import logging
from dataclasses import dataclass
from datetime import datetime


@dataclass
class AlphaGenomeConfig:
    """Configuration for AlphaGenome client."""
    base_url: str = "http://localhost:8000"
    timeout: int = 300
    max_retries: int = 3
    retry_delay: float = 1.0
    api_key: Optional[str] = None


class AlphaGenomeError(Exception):
    """Base exception for AlphaGenome client errors."""
    pass


class APIError(AlphaGenomeError):
    """Exception for API-related errors."""
    def __init__(self, message: str, status_code: int = None, response: dict = None):
        super().__init__(message)
        self.status_code = status_code
        self.response = response


class JobTimeoutError(AlphaGenomeError):
    """Exception for job timeout errors."""
    pass


class AlphaGenomeClient:
    """
    Client for interacting with AlphaGenome API.
    
    Provides high-level interface for:
    - Genomic track prediction
    - Variant effect scoring
    - Batch processing
    - Job management
    """
    
    def __init__(self, config: Optional[AlphaGenomeConfig] = None):
        """
        Initialize AlphaGenome client.
        
        Args:
            config: Client configuration
        """
        self.config = config or AlphaGenomeConfig()
        self.session = requests.Session()
        self.logger = logging.getLogger(__name__)
        
        # Set up authentication if API key provided
        if self.config.api_key:
            self.session.headers.update({
                'Authorization': f'Bearer {self.config.api_key}'
            })
        
        # Set default headers
        self.session.headers.update({
            'Content-Type': 'application/json',
            'User-Agent': 'AlphaGenome-Python-Client/1.0.0'
        })
    
    def _make_request(self, 
                     method: str,
                     endpoint: str,
                     data: dict = None,
                     params: dict = None,
                     retry_count: int = 0) -> requests.Response:
        """
        Make HTTP request with retry logic.
        
        Args:
            method: HTTP method
            endpoint: API endpoint
            data: Request data
            params: URL parameters
            retry_count: Current retry attempt
            
        Returns:
            HTTP response
            
        Raises:
            APIError: For API errors
        """
        url = f"{self.config.base_url.rstrip('/')}/{endpoint.lstrip('/')}"
        
        try:
            response = self.session.request(
                method=method,
                url=url,
                json=data,
                params=params,
                timeout=self.config.timeout
            )
            
            # Handle specific status codes
            if response.status_code == 200:
                return response
            elif response.status_code == 202:
                return response  # Accepted for async jobs
            elif response.status_code in [500, 502, 503, 504]:
                # Server errors - retry
                if retry_count < self.config.max_retries:
                    time.sleep(self.config.retry_delay * (2 ** retry_count))
                    return self._make_request(method, endpoint, data, params, retry_count + 1)
            
            # Try to parse error response
            try:
                error_data = response.json()
                error_message = error_data.get('detail', f'HTTP {response.status_code}')
            except:
                error_message = f'HTTP {response.status_code}: {response.text}'
            
            raise APIError(
                message=error_message,
                status_code=response.status_code,
                response=error_data if 'error_data' in locals() else None
            )
            
        except requests.exceptions.RequestException as e:
            if retry_count < self.config.max_retries:
                time.sleep(self.config.retry_delay * (2 ** retry_count))
                return self._make_request(method, endpoint, data, params, retry_count + 1)
            
            raise APIError(f"Request failed: {str(e)}")
    
    def health_check(self) -> Dict[str, Any]:
        """
        Check API health status.
        
        Returns:
            Health status information
        """
        response = self._make_request('GET', '/health')
        return response.json()
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the loaded model.
        
        Returns:
            Model information
        """
        response = self._make_request('GET', '/models/info')
        return response.json()
    
    def predict_tracks(self,
                      sequence: str,
                      organism: str = 'human',
                      tracks: Optional[List[str]] = None,
                      output_format: str = 'json') -> Dict[str, Any]:
        """
        Predict genomic tracks for a DNA sequence.
        
        Args:
            sequence: DNA sequence to analyze
            organism: Target organism ('human' or 'mouse')
            tracks: Specific tracks to predict (None for all)
            output_format: Output format ('json' or 'bigwig')
            
        Returns:
            Prediction results
            
        Raises:
            APIError: For API errors
            ValueError: For invalid parameters
        """
        # Validate inputs
        if not sequence:
            raise ValueError("Sequence cannot be empty")
        
        if len(sequence) > 1048576:
            raise ValueError("Sequence too long (max 1Mb)")
        
        if organism not in ['human', 'mouse']:
            raise ValueError("Organism must be 'human' or 'mouse'")
        
        # Prepare request
        request_data = {
            'sequence': sequence.upper(),
            'organism': organism,
            'output_format': output_format
        }
        
        if tracks:
            request_data['tracks'] = tracks
        
        # Make request
        response = self._make_request('POST', '/predict/tracks', data=request_data)
        result = response.json()
        
        self.logger.info(f"Predicted {result['metadata']['num_tracks']} tracks in {result['processing_time']:.2f}s")
        
        return result
    
    def score_variant(self,
                     chromosome: str,
                     position: int,
                     ref_allele: str,
                     alt_allele: str,
                     variant_id: Optional[str] = None,
                     organism: str = 'human') -> Dict[str, Any]:
        """
        Score a single genetic variant.
        
        Args:
            chromosome: Chromosome name
            position: Variant position (0-based)
            ref_allele: Reference allele
            alt_allele: Alternative allele
            variant_id: Optional variant identifier
            organism: Target organism
            
        Returns:
            Variant score results
        """
        # Prepare request
        request_data = {
            'chromosome': chromosome,
            'position': position,
            'ref_allele': ref_allele.upper(),
            'alt_allele': alt_allele.upper(),
            'organism': organism
        }
        
        if variant_id:
            request_data['variant_id'] = variant_id
        
        # Make request
        response = self._make_request('POST', '/score/variant', data=request_data)
        result = response.json()
        
        self.logger.info(f"Scored variant {variant_id or 'unnamed'} in {result['processing_time']:.2f}s")
        
        return result
    
    def score_variants_batch(self,
                           variants: List[Dict[str, Any]],
                           organism: str = 'human',
                           scorers: Optional[List[str]] = None,
                           wait_for_completion: bool = True,
                           polling_interval: float = 5.0,
                           timeout: Optional[float] = None) -> Union[str, Dict[str, Any]]:
        """
        Score multiple variants in batch.
        
        Args:
            variants: List of variant dictionaries
            organism: Target organism
            scorers: Specific scorers to use
            wait_for_completion: Whether to wait for job completion
            polling_interval: How often to check job status (seconds)
            timeout: Maximum time to wait for completion
            
        Returns:
            Job ID if not waiting, or results if waiting for completion
        """
        # Prepare request
        request_data = {
            'variants': variants,
            'organism': organism
        }
        
        if scorers:
            request_data['scorers'] = scorers
        
        # Submit batch job
        response = self._make_request('POST', '/score/variants/batch', data=request_data)
        job_info = response.json()
        job_id = job_info['job_id']
        
        self.logger.info(f"Submitted batch job {job_id} with {len(variants)} variants")
        
        if not wait_for_completion:
            return job_id
        
        # Wait for completion
        start_time = time.time()
        
        while True:
            job_status = self.get_job_status(job_id)
            
            if job_status['status'] == 'completed':
                self.logger.info(f"Batch job {job_id} completed successfully")
                return job_status['result']
            
            elif job_status['status'] == 'failed':
                error_msg = job_status.get('error', 'Unknown error')
                raise APIError(f"Batch job {job_id} failed: {error_msg}")
            
            # Check timeout
            if timeout and (time.time() - start_time) > timeout:
                raise JobTimeoutError(f"Job {job_id} timed out after {timeout} seconds")
            
            # Wait before next check
            time.sleep(polling_interval)
    
    def get_job_status(self, job_id: str) -> Dict[str, Any]:
        """
        Get status of a background job.
        
        Args:
            job_id: Job identifier
            
        Returns:
            Job status information
        """
        response = self._make_request('GET', f'/jobs/{job_id}')
        return response.json()
    
    def score_variants_from_vcf(self,
                               vcf_file: str,
                               organism: str = 'human',
                               chromosome: Optional[str] = None,
                               start: Optional[int] = None,
                               end: Optional[int] = None,
                               max_variants: Optional[int] = None,
                               output_file: Optional[str] = None) -> Union[pd.DataFrame, str]:
        """
        Score variants from a VCF file.
        
        Args:
            vcf_file: Path to VCF file
            organism: Target organism
            chromosome: Filter by chromosome
            start: Start position filter
            end: End position filter
            max_variants: Maximum number of variants to process
            output_file: Optional output file path
            
        Returns:
            DataFrame with results or job ID
        """
        # Load variants from VCF
        variants = self._load_variants_from_vcf(
            vcf_file, chromosome, start, end, max_variants
        )
        
        if not variants:
            raise ValueError("No variants found in VCF file")
        
        self.logger.info(f"Loaded {len(variants)} variants from {vcf_file}")
        
        # Score variants in batch
        results = self.score_variants_batch(
            variants=variants,
            organism=organism,
            wait_for_completion=True
        )
        
        # Convert to DataFrame
        df = self._results_to_dataframe(results['results'])
        
        # Save to file if requested
        if output_file:
            df.to_csv(output_file, index=False, sep='\t')
            self.logger.info(f"Results saved to {output_file}")
            return output_file
        
        return df
    
    def _load_variants_from_vcf(self,
                              vcf_file: str,
                              chromosome: Optional[str] = None,
                              start: Optional[int] = None,
                              end: Optional[int] = None,
                              max_variants: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Load variants from VCF file.
        
        Args:
            vcf_file: Path to VCF file
            chromosome: Filter by chromosome
            start: Start position filter
            end: End position filter
            max_variants: Maximum variants to load
            
        Returns:
            List of variant dictionaries
        """
        try:
            import pysam
        except ImportError:
            raise ImportError("pysam is required for VCF processing. Install with: pip install pysam")
        
        variants = []
        
        with pysam.VariantFile(vcf_file) as vcf:
            if chromosome:
                region = f"{chromosome}:{start}-{end}" if start and end else chromosome
                records = vcf.fetch(region)
            else:
                records = vcf.fetch()
            
            for i, record in enumerate(records):
                if max_variants and i >= max_variants:
                    break
                
                variant = {
                    'chromosome': record.chrom,
                    'position': record.pos - 1,  # Convert to 0-based
                    'ref_allele': record.ref,
                    'alt_allele': str(record.alts[0]),  # Take first alternative
                }
                
                if record.id:
                    variant['variant_id'] = record.id
                
                variants.append(variant)
        
        return variants
    
    def _results_to_dataframe(self, results: List[Dict[str, Any]]) -> pd.DataFrame:
        """
        Convert batch results to pandas DataFrame.
        
        Args:
            results: List of variant score results
            
        Returns:
            DataFrame with results
        """
        rows = []
        
        for result in results:
            variant = result['variant']
            scores = result['scores']
            metadata = result['metadata']
            
            row = {
                'variant_id': variant.get('variant_id'),
                'chromosome': variant['chromosome'],
                'position': variant['position'],
                'ref_allele': variant['ref_allele'],
                'alt_allele': variant['alt_allele'],
            }
            
            # Add all scores as columns
            row.update(scores)
            
            # Add error status if present
            if 'error' in metadata:
                row['error'] = metadata['error']
                row['status'] = 'failed'
            else:
                row['status'] = 'success'
            
            rows.append(row)
        
        return pd.DataFrame(rows)
    
    def create_analysis_report(self,
                             results_df: pd.DataFrame,
                             output_dir: str = "alphagenome_analysis") -> str:
        """
        Create comprehensive analysis report from variant scoring results.
        
        Args:
            results_df: DataFrame with variant scoring results
            output_dir: Output directory for report
            
        Returns:
            Path to generated report
        """
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Generate summary statistics
        summary = self._generate_summary_stats(results_df)
        
        # Create visualizations
        plots = self._create_visualizations(results_df, output_path)
        
        # Generate HTML report
        report_path = output_path / "alphagenome_report.html"
        self._generate_html_report(summary, plots, report_path)
        
        self.logger.info(f"Analysis report generated: {report_path}")
        return str(report_path)
    
    def _generate_summary_stats(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate summary statistics from results.
        
        Args:
            df: Results DataFrame
            
        Returns:
            Summary statistics
        """
        stats = {
            'total_variants': len(df),
            'successful': len(df[df['status'] == 'success']),
            'failed': len(df[df['status'] == 'failed']),
            'chromosomes': df['chromosome'].unique().tolist(),
        }
        
        # Score statistics
        score_columns = [col for col in df.columns if col.endswith('_effect') or col.endswith('_mse')]
        
        if score_columns:
            stats['score_stats'] = {}
            for col in score_columns:
                if df[col].dtype in ['float64', 'int64']:
                    stats['score_stats'][col] = {
                        'mean': float(df[col].mean()),
                        'std': float(df[col].std()),
                        'min': float(df[col].min()),
                        'max': float(df[col].max())
                    }
        
        return stats
    
    def _create_visualizations(self, df: pd.DataFrame, output_path: Path) -> List[str]:
        """
        Create visualizations from results data.
        
        Args:
            df: Results DataFrame
            output_path: Output directory
            
        Returns:
            List of generated plot files
        """
        plots = []
        
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            # Set style
            plt.style.use('default')
            sns.set_palette("husl")
            
            # Plot 1: Variant distribution by chromosome
            if 'chromosome' in df.columns:
                plt.figure(figsize=(12, 6))
                chrom_counts = df['chromosome'].value_counts().sort_index()
                chrom_counts.plot(kind='bar')
                plt.title('Variant Distribution by Chromosome')
                plt.xlabel('Chromosome')
                plt.ylabel('Number of Variants')
                plt.xticks(rotation=45)
                plt.tight_layout()
                
                plot_path = output_path / 'chromosome_distribution.png'
                plt.savefig(plot_path, dpi=300, bbox_inches='tight')
                plots.append(str(plot_path))
                plt.close()
            
            # Plot 2: Score distributions
            score_columns = [col for col in df.columns if col.endswith('_effect')]
            if score_columns:
                fig, axes = plt.subplots(2, 2, figsize=(15, 10))
                axes = axes.flatten()
                
                for i, col in enumerate(score_columns[:4]):
                    if i < len(axes) and df[col].dtype in ['float64', 'int64']:
                        df[col].hist(bins=50, ax=axes[i], alpha=0.7)
                        axes[i].set_title(f'{col} Distribution')
                        axes[i].set_xlabel('Score')
                        axes[i].set_ylabel('Frequency')
                plt.tight_layout()
                plot_path = output_path / 'score_distributions.png'
                plt.savefig(plot_path, dpi=300, bbox_inches='tight')
                plots.append(str(plot_path))
                plt.close()
           
        except ImportError:
            self.logger.warning("Matplotlib/seaborn not available. Skipping visualizations.")
        
        return plots
    
    def _generate_html_report(self, summary: Dict[str, Any], plots: List[str], output_path: Path):
        """
        Generate HTML report.
        
        Args:
            summary: Summary statistics
            plots: List of plot file paths
            output_path: Output file path
        """
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>AlphaGenome Analysis Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                .header {{ text-align: center; margin-bottom: 40px; }}
                .section {{ margin-bottom: 30px; }}
                .stats-table {{ border-collapse: collapse; width: 100%; }}
                .stats-table th, .stats-table td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                .stats-table th {{ background-color: #f2f2f2; }}
                .plot {{ text-align: center; margin: 20px 0; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>AlphaGenome Variant Analysis Report</h1>
                <p>Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>
            
            <div class="section">
                <h2>Summary Statistics</h2>
                <table class="stats-table">
                    <tr><th>Metric</th><th>Value</th></tr>
                    <tr><td>Total Variants</td><td>{summary['total_variants']}</td></tr>
                    <tr><td>Successfully Scored</td><td>{summary['successful']}</td></tr>
                    <tr><td>Failed</td><td>{summary['failed']}</td></tr>
                    <tr><td>Success Rate</td><td>{summary['successful']/summary['total_variants']*100:.1f}%</td></tr>
                    <tr><td>Chromosomes</td><td>{', '.join(summary['chromosomes'])}</td></tr>
                </table>
            </div>
        """
        
        # Add plots
        if plots:
            html_content += '<div class="section"><h2>Visualizations</h2>'
            for plot_path in plots:
                plot_name = Path(plot_path).name
                html_content += f'<div class="plot"><img src="{plot_name}" alt="{plot_name}" style="max-width: 800px;"></div>'
            html_content += '</div>'
        
        html_content += """
        </body>
        </html>
        """
        
        with open(output_path, 'w') as f:
            f.write(html_content)


# Convenience functions
def quick_predict(sequence: str, organism: str = 'human', **kwargs) -> Dict[str, Any]:
    """
    Quick prediction with default client.
    
    Args:
        sequence: DNA sequence
        organism: Target organism
        **kwargs: Additional arguments for client
        
    Returns:
        Prediction results
    """
    client = AlphaGenomeClient()
    return client.predict_tracks(sequence, organism, **kwargs)


def quick_score_variant(chromosome: str, position: int, ref: str, alt: str, 
                       organism: str = 'human', **kwargs) -> Dict[str, Any]:
    """
    Quick variant scoring with default client.
    
    Args:
        chromosome: Chromosome name
        position: Variant position
        ref: Reference allele
        alt: Alternative allele
        organism: Target organism
        **kwargs: Additional arguments
        
    Returns:
        Variant score results
    """
    client = AlphaGenomeClient()
    return client.score_variant(chromosome, position, ref, alt, organism=organism, **kwargs)