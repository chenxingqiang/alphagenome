"""
AlphaGenome API Server

FastAPI-based REST API for AlphaGenome model inference:
- Genomic track prediction endpoints
- Variant scoring endpoints
- Batch processing support
- Model management and health checks
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any, Union
import uvicorn
import logging
import asyncio
import pickle
import hashlib
from pathlib import Path
import time
import traceback
from datetime import datetime

from ..core.model import create_alphagenome_model
from ..core.data_processor import DNAEncoder
from ..core.variant_scorer import VariantScorer, Variant, VariantScore
from ..core.trainer import AlphaGenomeTrainer


# Pydantic models for API requests/responses
class PredictTracksRequest(BaseModel):
    """Request model for track prediction."""
    sequence: str = Field(..., description="DNA sequence to analyze", max_length=1048576)
    organism: str = Field("human", description="Target organism (human/mouse)")
    tracks: Optional[List[str]] = Field(None, description="Specific tracks to predict")
    output_format: str = Field("json", description="Output format (json/bigwig)")
    
    class Config:
        schema_extra = {
            "example": {
                "sequence": "ATCGATCGATCG" * 100,
                "organism": "human",
                "tracks": ["rna_seq", "atac_seq"],
                "output_format": "json"
            }
        }


class PredictTracksResponse(BaseModel):
    """Response model for track prediction."""
    predictions: Dict[str, List[float]]
    metadata: Dict[str, Any]
    processing_time: float


class VariantRequest(BaseModel):
    """Request model for single variant scoring."""
    chromosome: str = Field(..., description="Chromosome name")
    position: int = Field(..., description="Variant position (0-based)")
    ref_allele: str = Field(..., description="Reference allele")
    alt_allele: str = Field(..., description="Alternative allele")
    variant_id: Optional[str] = Field(None, description="Variant identifier")
    organism: str = Field("human", description="Target organism")
    
    class Config:
        schema_extra = {
            "example": {
                "chromosome": "chr1",
                "position": 1000000,
                "ref_allele": "A",
                "alt_allele": "G",
                "variant_id": "rs123456",
                "organism": "human"
            }
        }


class BatchVariantRequest(BaseModel):
    """Request model for batch variant scoring."""
    variants: List[VariantRequest]
    organism: str = Field("human", description="Target organism")
    scorers: Optional[List[str]] = Field(None, description="Specific scorers to use")
    
    class Config:
        schema_extra = {
            "example": {
                "variants": [
                    {
                        "chromosome": "chr1",
                        "position": 1000000,
                        "ref_allele": "A",
                        "alt_allele": "G",
                        "variant_id": "rs123456"
                    }
                ],
                "organism": "human",
                "scorers": ["RNA-seq", "ATAC-seq"]
            }
        }


class VariantScoreResponse(BaseModel):
    """Response model for variant scoring."""
    variant: Dict[str, Any]
    scores: Dict[str, float]
    metadata: Dict[str, Any]
    processing_time: float


class JobStatusResponse(BaseModel):
    """Response model for job status."""
    job_id: str
    status: str  # pending, running, completed, failed
    progress: Optional[float] = None
    result: Optional[Any] = None
    error: Optional[str] = None
    created_at: datetime
    updated_at: datetime


class HealthResponse(BaseModel):
    """Response model for health check."""
    status: str
    version: str
    model_loaded: bool
    uptime: float
    memory_usage: Dict[str, float]


# Background job management
class JobManager:
    """Manages background jobs for batch processing."""
    
    def __init__(self):
        self.jobs = {}
        self.job_counter = 0
    
    def create_job(self, job_type: str) -> str:
        """Create a new job and return job ID."""
        self.job_counter += 1
        job_id = f"{job_type}_{self.job_counter}_{int(time.time())}"
        
        self.jobs[job_id] = {
            'id': job_id,
            'type': job_type,
            'status': 'pending',
            'progress': 0.0,
            'result': None,
            'error': None,
            'created_at': datetime.utcnow(),
            'updated_at': datetime.utcnow()
        }
        
        return job_id
    
    def update_job(self, job_id: str, status: str = None, progress: float = None, 
                   result: Any = None, error: str = None):
        """Update job status."""
        if job_id in self.jobs:
            job = self.jobs[job_id]
            if status:
                job['status'] = status
            if progress is not None:
                job['progress'] = progress
            if result is not None:
                job['result'] = result
            if error:
                job['error'] = error
            job['updated_at'] = datetime.utcnow()
    
    def get_job(self, job_id: str) -> Optional[Dict]:
        """Get job information."""
        return self.jobs.get(job_id)
    
    def cleanup_old_jobs(self, max_age_hours: int = 24):
        """Clean up jobs older than max_age_hours."""
        current_time = datetime.utcnow()
        to_remove = []
        
        for job_id, job in self.jobs.items():
            age = (current_time - job['created_at']).total_seconds() / 3600
            if age > max_age_hours:
                to_remove.append(job_id)
        
        for job_id in to_remove:
            del self.jobs[job_id]


# Global variables
app = FastAPI(
    title="AlphaGenome API",
    description="REST API for AlphaGenome genomic sequence modeling",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state
model = None
model_params = None
variant_scorer = None
job_manager = JobManager()
server_start_time = time.time()
logger = logging.getLogger(__name__)

# Model configuration
MODEL_CONFIG = {
    'sequence_length': 1048576,
    'model_dim': 1536,
    'num_heads': 16,
    'num_transformer_layers': 9,
    'organisms': ['human', 'mouse'],
    'reference_genomes': {
        'human': 'data/hg38.fa',
        'mouse': 'data/mm10.fa'
    }
}


@app.on_event("startup")
async def startup_event():
    """Initialize model and services on startup."""
    global model, model_params, variant_scorer
    
    logger.info("Starting AlphaGenome API server...")
    
    try:
        # Initialize model
        model = create_alphagenome_model(MODEL_CONFIG)
        
        # Load pre-trained parameters (would be actual checkpoint in production)
        checkpoint_path = "checkpoints/alphagenome_student_model.pkl"
        if Path(checkpoint_path).exists():
            with open(checkpoint_path, 'rb') as f:
                checkpoint = pickle.load(f)
                model_params = checkpoint['params']
        else:
            logger.warning("No checkpoint found, using random parameters")
            import jax
            import jax.numpy as jnp
            key = jax.random.PRNGKey(42)
            dummy_input = jnp.ones((1, 1048576, 4))
            dummy_organism = jnp.array([0])
            model_params = model.init(key, dummy_input, dummy_organism)
        
        # Initialize variant scorer
        variant_scorer = VariantScorer(model, model_params, MODEL_CONFIG)
        
        logger.info("AlphaGenome API server started successfully")
        
    except Exception as e:
        logger.error(f"Failed to start server: {e}")
        raise


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    logger.info("Shutting down AlphaGenome API server...")


# Dependency injection
def get_model():
    """Dependency to get model instance."""
    if model is None or model_params is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded"
        )
    return model, model_params


def get_variant_scorer():
    """Dependency to get variant scorer instance."""
    if variant_scorer is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Variant scorer not loaded"
        )
    return variant_scorer


# API endpoints
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    import psutil
    
    uptime = time.time() - server_start_time
    process = psutil.Process()
    memory_info = process.memory_info()
    
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        model_loaded=model is not None and model_params is not None,
        uptime=uptime,
        memory_usage={
            "rss": memory_info.rss / 1024 / 1024,  # MB
            "vms": memory_info.vms / 1024 / 1024,  # MB
            "percent": process.memory_percent()
        }
    )


@app.post("/predict/tracks", response_model=PredictTracksResponse)
async def predict_tracks(
    request: PredictTracksRequest,
    model_data=Depends(get_model)
):
    """Predict genomic tracks for a DNA sequence."""
    start_time = time.time()
    model, model_params = model_data
    
    try:
        # Validate sequence
        if not request.sequence or len(request.sequence) > 1048576:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Sequence must be non-empty and ≤ 1Mb"
            )
        
        # Validate organism
        if request.organism not in ['human', 'mouse']:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Organism must be 'human' or 'mouse'"
            )
        
        # Encode sequence
        encoder = DNAEncoder()
        encoded_seq = encoder.encode_sequence(request.sequence)
        
        # Make prediction
        import jax.numpy as jnp
        batch_seq = jnp.array(encoded_seq)[None, :, :]
        organism_id = jnp.array([0 if request.organism == 'human' else 1])
        
        predictions = model.apply(model_params, batch_seq, organism_id)
        
        # Convert predictions to lists for JSON serialization
        output_predictions = {}
        for key, pred in predictions.items():
            if request.tracks is None or key in request.tracks:
                # Convert to numpy and then to list
                pred_array = pred[0]  # Remove batch dimension
                if len(pred_array.shape) == 1:
                    output_predictions[key] = pred_array.tolist()
                elif len(pred_array.shape) == 2:
                    output_predictions[key] = pred_array.tolist()
                else:
                    # For 3D arrays, flatten or summarize
                    output_predictions[key] = pred_array.flatten().tolist()
        
        processing_time = time.time() - start_time
        
        metadata = {
            'sequence_length': len(request.sequence),
            'organism': request.organism,
            'num_tracks': len(output_predictions),
            'model_version': '1.0.0'
        }
        
        return PredictTracksResponse(
            predictions=output_predictions,
            metadata=metadata,
            processing_time=processing_time
        )
        
    except Exception as e:
        logger.error(f"Track prediction error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}"
        )


@app.post("/score/variant", response_model=VariantScoreResponse)
async def score_variant(
    request: VariantRequest,
    scorer=Depends(get_variant_scorer)
):
    """Score a single genetic variant."""
    start_time = time.time()
    
    try:
        # Create variant object
        variant = Variant(
            chromosome=request.chromosome,
            position=request.position,
            ref_allele=request.ref_allele,
            alt_allele=request.alt_allele,
            variant_id=request.variant_id
        )
        
        # Get reference genome path
        ref_genome = MODEL_CONFIG['reference_genomes'][request.organism]
        
        # Score variant
        variant_score = scorer.score_variant(variant, ref_genome, request.organism)
        
        processing_time = time.time() - start_time
        
        return VariantScoreResponse(
            variant={
                'chromosome': variant_score.variant.chromosome,
                'position': variant_score.variant.position,
                'ref_allele': variant_score.variant.ref_allele,
                'alt_allele': variant_score.variant.alt_allele,
                'variant_id': variant_score.variant.variant_id
            },
            scores=variant_score.scores,
            metadata={
                **variant_score.metadata,
                'model_version': '1.0.0'
            },
            processing_time=processing_time
        )
        
    except Exception as e:
        logger.error(f"Variant scoring error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Variant scoring failed: {str(e)}"
        )


@app.post("/score/variants/batch")
async def score_variants_batch(
    request: BatchVariantRequest,
    background_tasks: BackgroundTasks,
    scorer=Depends(get_variant_scorer)
):
    """Submit batch variant scoring job."""
    try:
        # Create job
        job_id = job_manager.create_job("batch_variant_scoring")
        
        # Add background task
        background_tasks.add_task(
            process_batch_variants,
            job_id,
            request,
            scorer
        )
        
        return JSONResponse(
            status_code=status.HTTP_202_ACCEPTED,
            content={
                "job_id": job_id,
                "status": "submitted",
                "message": f"Batch job submitted with {len(request.variants)} variants"
            }
        )
        
    except Exception as e:
        logger.error(f"Batch submission error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch submission failed: {str(e)}"
        )


async def process_batch_variants(job_id: str, request: BatchVariantRequest, scorer):
    """Process batch variant scoring in background."""
    try:
        job_manager.update_job(job_id, status="running", progress=0.0)
        
        # Convert request variants to Variant objects
        variants = []
        for var_req in request.variants:
            variant = Variant(
                chromosome=var_req.chromosome,
                position=var_req.position,
                ref_allele=var_req.ref_allele,
                alt_allele=var_req.alt_allele,
                variant_id=var_req.variant_id
            )
            variants.append(variant)
        
        # Get reference genome
        ref_genome = MODEL_CONFIG['reference_genomes'][request.organism]
        
        # Process variants
        results = []
        total_variants = len(variants)
        
        for i, variant in enumerate(variants):
            try:
                variant_score = scorer.score_variant(variant, ref_genome, request.organism)
                
                result = {
                    'variant': {
                        'chromosome': variant_score.variant.chromosome,
                        'position': variant_score.variant.position,
                        'ref_allele': variant_score.variant.ref_allele,
                        'alt_allele': variant_score.variant.alt_allele,
                        'variant_id': variant_score.variant.variant_id
                    },
                    'scores': variant_score.scores,
                    'metadata': variant_score.metadata
                }
                results.append(result)
                
                # Update progress
                progress = (i + 1) / total_variants
                job_manager.update_job(job_id, progress=progress)
                
            except Exception as e:
                logger.error(f"Failed to score variant {variant.variant_id}: {e}")
                # Add error result
                results.append({
                    'variant': {
                        'chromosome': variant.chromosome,
                        'position': variant.position,
                        'ref_allele': variant.ref_allele,
                        'alt_allele': variant.alt_allele,
                        'variant_id': variant.variant_id
                    },
                    'scores': {},
                    'metadata': {'error': str(e)}
                })
        
        # Complete job
        job_manager.update_job(
            job_id,
            status="completed",
            progress=1.0,
            result={
                'results': results,
                'summary': {
                    'total_variants': total_variants,
                    'successful': len([r for r in results if 'error' not in r['metadata']]),
                    'failed': len([r for r in results if 'error' in r['metadata']])
                }
            }
        )
        
    except Exception as e:
        logger.error(f"Batch processing error for job {job_id}: {e}")
        job_manager.update_job(
            job_id,
            status="failed",
            error=str(e)
        )


@app.get("/jobs/{job_id}", response_model=JobStatusResponse)
async def get_job_status(job_id: str):
    """Get status of a background job."""
    job = job_manager.get_job(job_id)
    
    if job is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Job not found"
        )
    
    return JobStatusResponse(
        job_id=job['id'],
        status=job['status'],
        progress=job['progress'],
        result=job['result'],
        error=job['error'],
        created_at=job['created_at'],
        updated_at=job['updated_at']
    )


@app.get("/models/info")
async def get_model_info():
    """Get information about loaded model."""
    return {
        "model_loaded": model is not None and model_params is not None,
        "config": MODEL_CONFIG,
        "supported_organisms": ["human", "mouse"],
        "max_sequence_length": 1048576,
        "available_tracks": [
            "rna_seq_1bp", "rna_seq_128bp",
            "atac_seq_1bp", "atac_seq_128bp", 
            "dnase_seq_1bp", "dnase_seq_128bp",
            "cage_1bp", "cage_128bp",
            "pro_cap_1bp", "pro_cap_128bp",
            "chip_tf", "chip_histone",
            "contact_maps",
            "splice_sites", "splice_usage", "splice_junctions"
        ]
    }


# Cleanup task
@app.on_event("startup")
async def start_cleanup_task():
    """Start periodic cleanup task."""
    async def cleanup_jobs():
        while True:
            try:
                job_manager.cleanup_old_jobs(max_age_hours=24)
                await asyncio.sleep(3600)  # Run every hour
            except Exception as e:
                logger.error(f"Cleanup task error: {e}")
                await asyncio.sleep(3600)
    
    asyncio.create_task(cleanup_jobs())


# Custom error handlers
@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Handle general exceptions."""
    logger.error(f"Unhandled exception: {exc}")
    logger.error(traceback.format_exc())
    
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "detail": "Internal server error",
            "type": type(exc).__name__
        }
    )


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run server
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )