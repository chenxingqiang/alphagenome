#!/usr/bin/env python3
"""
AlphaGenome Training Script

Complete training pipeline for AlphaGenome model including:
- Data preprocessing and validation
- Multi-stage training (pretraining + distillation)
- Model evaluation and checkpointing
- Variant scorer calibration

Usage:
    python train_alphagenome.py --config config.yaml
    python train_alphagenome.py --stage pretrain --organism human --fold 0
    python train_alphagenome.py --stage distill --checkpoint_dir ./checkpoints
"""

import argparse
import logging
import yaml
import sys
from pathlib import Path
import time
from typing import Dict, Any, List

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from alphagenome.core.trainer import AlphaGenomeTrainer, train_full_pipeline
from alphagenome.core.data_processor import AlphaGenomeDataProcessor
from alphagenome.core.variant_scorer import VariantScorer
from alphagenome.core.model import create_alphagenome_model


def setup_logging(log_level: str = "INFO", log_file: str = None):
    """Setup logging configuration."""
    level = getattr(logging, log_level.upper())
    
    handlers = [logging.StreamHandler()]
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=handlers
    )


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Set default values
    defaults = {
        'sequence_length': 1048576,
        'model_dim': 1536,
        'num_heads': 16,
        'num_transformer_layers': 9,
        'learning_rate': 0.004,
        'weight_decay': 0.4,
        'batch_size': 64,
        'num_steps': 15000,
        'warmup_steps': 5000,
        'checkpoint_dir': './checkpoints',
        'data_dir': './data',
        'organisms': ['human', 'mouse'],
        'reference_genomes': {
            'human': './data/hg38.fa',
            'mouse': './data/mm10.fa'
        }
    }
    
    # Merge with defaults
    for key, value in defaults.items():
        if key not in config:
            config[key] = value
    
    return config


def validate_data_paths(config: Dict[str, Any]) -> bool:
    """Validate that required data files exist."""
    logger = logging.getLogger(__name__)
    
    # Check reference genomes
    for organism, genome_path in config['reference_genomes'].items():
        if not Path(genome_path).exists():
            logger.error(f"Reference genome not found: {genome_path}")
            return False
    
    # Check data directories
    data_dir = Path(config['data_dir'])
    if not data_dir.exists():
        logger.error(f"Data directory not found: {data_dir}")
        return False
    
    # Check for at least some data files
    required_dirs = ['encode', 'gtex', 'fantom5', '4dn']
    for req_dir in required_dirs:
        if not (data_dir / req_dir).exists():
            logger.warning(f"Recommended data directory missing: {data_dir / req_dir}")
    
    return True


def pretrain_stage(config: Dict[str, Any], organism: str, fold: int) -> str:
    """Run pretraining stage for specific organism and fold."""
    logger = logging.getLogger(__name__)
    logger.info(f"Starting pretraining for {organism} fold {fold}")
    
    # Initialize trainer
    trainer = AlphaGenomeTrainer(config)
    
    # Run pretraining
    start_time = time.time()
    params = trainer.pretrain_fold(organism, fold)
    training_time = time.time() - start_time
    
    logger.info(f"Pretraining completed in {training_time:.2f} seconds")
    
    # Evaluate model
    metrics = trainer.evaluate_model(params, organism, fold, split='valid')
    logger.info(f"Validation metrics: {metrics}")
    
    # Save final checkpoint
    checkpoint_path = Path(config['checkpoint_dir']) / f"{organism}_fold_{fold}_final.pkl"
    trainer._save_checkpoint(params, None, organism, fold, config['num_steps'])
    
    return str(checkpoint_path)


def train_teacher_models(config: Dict[str, Any], organism: str, num_teachers: int = 64) -> List[str]:
    """Train ensemble of teacher models."""
    logger = logging.getLogger(__name__)
    logger.info(f"Training {num_teachers} teacher models for {organism}")
    
    trainer = AlphaGenomeTrainer(config)
    teacher_checkpoints = []
    
    for i in range(num_teachers):
        logger.info(f"Training teacher model {i+1}/{num_teachers}")
        
        # Train all-folds model
        params = trainer.train_all_folds_model(organism)
        
        # Save checkpoint
        checkpoint_path = Path(config['checkpoint_dir']) / f"{organism}_teacher_{i}.pkl"
        trainer._save_checkpoint(params, None, organism, -1, config['num_steps'])
        teacher_checkpoints.append(str(checkpoint_path))
    
    return teacher_checkpoints


def distillation_stage(config: Dict[str, Any], teacher_checkpoints: List[str]) -> str:
    """Run distillation stage."""
    logger = logging.getLogger(__name__)
    logger.info("Starting distillation training")
    
    # Load teacher models
    teacher_params_list = []
    for checkpoint_path in teacher_checkpoints:
        trainer = AlphaGenomeTrainer(config)
        params, _, _ = trainer.load_checkpoint(checkpoint_path)
        teacher_params_list.append(params)
    
    # Run distillation
    trainer = AlphaGenomeTrainer(config)
    student_params = trainer.distill_student_model(teacher_params_list)
    
    # Save student model
    checkpoint_path = Path(config['checkpoint_dir']) / "alphagenome_student_model.pkl"
    trainer._save_checkpoint(student_params, None, "student", 0, 250000)
    
    logger.info(f"Distillation completed. Student model saved to {checkpoint_path}")
    return str(checkpoint_path)


def evaluate_model(config: Dict[str, Any], checkpoint_path: str):
    """Comprehensive model evaluation."""
    logger = logging.getLogger(__name__)
    logger.info("Running comprehensive model evaluation")
    
    # Load model
    trainer = AlphaGenomeTrainer(config)
    params, _, _ = trainer.load_checkpoint(checkpoint_path)
    
    # Evaluate on all test sets
    results = {}
    
    for organism in ['human', 'mouse']:
        results[organism] = {}
        
        for fold in range(4):
            # Evaluate on test set
            test_metrics = trainer.evaluate_model(params, organism, fold, split='test')
            results[organism][fold] = test_metrics
            
            logger.info(f"{organism} fold {fold} test metrics: {test_metrics}")
    
    # Save evaluation results
    import json
    results_path = Path(config['checkpoint_dir']) / "evaluation_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Evaluation results saved to {results_path}")


def calibrate_variant_scorers(config: Dict[str, Any], checkpoint_path: str):
    """Calibrate variant scorers using held-out data."""
    logger = logging.getLogger(__name__)
    logger.info("Calibrating variant scorers")
    
    # Load model
    trainer = AlphaGenomeTrainer(config)
    params, _, _ = trainer.load_checkpoint(checkpoint_path)
    
    # Create variant scorer
    model = create_alphagenome_model(config)
    scorer = VariantScorer(model, params, config)
    
    # Generate calibration dataset (simplified - would use real variants)
    logger.info("Generating calibration dataset...")
    
    # Create recommended scorers
    recommended_scorers = scorer.create_recommended_scorers()
    
    logger.info(f"Created {len(recommended_scorers)} recommended variant scorers")
    
    # Save calibrated scorers
    import pickle
    scorers_path = Path(config['checkpoint_dir']) / "variant_scorers.pkl"
    with open(scorers_path, 'wb') as f:
        pickle.dump(recommended_scorers, f)
    
    logger.info(f"Calibrated scorers saved to {scorers_path}")


def create_demo_config() -> Dict[str, Any]:
    """Create demo configuration for testing."""
    return {
        'sequence_length': 1048576,
        'model_dim': 1536,
        'num_heads': 16,
        'num_transformer_layers': 9,
        'learning_rate': 0.004,
        'weight_decay': 0.4,
        'batch_size': 2,  # Small batch for demo
        'num_steps': 100,  # Few steps for demo
        'warmup_steps': 50,
        'checkpoint_dir': './demo_checkpoints',
        'data_dir': './demo_data',
        'organisms': ['human'],
        'reference_genomes': {
            'human': './demo_data/demo_hg38.fa'
        },
        'human_rna_seq_files': [],
        'human_atac_files': [],
        'human_dnase_files': [],
        'human_chip_files': [],
        'human_contact_files': []
    }


def main():
    """Main training script."""
    parser = argparse.ArgumentParser(description="AlphaGenome Training Pipeline")
    parser.add_argument("--config", type=str, help="Path to configuration file")
    parser.add_argument("--stage", type=str, choices=['pretrain', 'teacher', 'distill', 'evaluate', 'calibrate', 'full'],
                       default='full', help="Training stage to run")
    parser.add_argument("--organism", type=str, choices=['human', 'mouse'], help="Target organism")
    parser.add_argument("--fold", type=int, choices=[0, 1, 2, 3], help="Cross-validation fold")
    parser.add_argument("--checkpoint-dir", type=str, default="./checkpoints", help="Checkpoint directory")
    parser.add_argument("--log-level", type=str, default="INFO", help="Logging level")
    parser.add_argument("--log-file", type=str, help="Log file path")
    parser.add_argument("--demo", action="store_true", help="Run with demo configuration")
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level, args.log_file)
    logger = logging.getLogger(__name__)
    
    # Load configuration
    if args.demo:
        config = create_demo_config()
        config['checkpoint_dir'] = args.checkpoint_dir
        logger.info("Using demo configuration")
    elif args.config:
        config = load_config(args.config)
        logger.info(f"Loaded configuration from {args.config}")
    else:
        logger.error("Must provide either --config or --demo")
        return 1
    
    # Create checkpoint directory
    Path(config['checkpoint_dir']).mkdir(exist_ok=True)
    
    # Validate data paths (skip for demo)
    if not args.demo and not validate_data_paths(config):
        logger.error("Data validation failed")
        return 1
    
    try:
        if args.stage == 'pretrain':
            if not args.organism or args.fold is None:
                logger.error("Pretraining requires --organism and --fold")
                return 1
            
            checkpoint_path = pretrain_stage(config, args.organism, args.fold)
            logger.info(f"Pretraining completed. Checkpoint: {checkpoint_path}")
        
        elif args.stage == 'teacher':
            if not args.organism:
                logger.error("Teacher training requires --organism")
                return 1
            
            checkpoints = train_teacher_models(config, args.organism)
            logger.info(f"Teacher training completed. {len(checkpoints)} models trained")
        
        elif args.stage == 'distill':
            # Find teacher checkpoints
            checkpoint_dir = Path(config['checkpoint_dir'])
            teacher_checkpoints = list(checkpoint_dir.glob("*_teacher_*.pkl"))
            
            if not teacher_checkpoints:
                logger.error("No teacher checkpoints found")
                return 1
            
            student_checkpoint = distillation_stage(config, [str(cp) for cp in teacher_checkpoints])
            logger.info(f"Distillation completed. Student model: {student_checkpoint}")
        
        elif args.stage == 'evaluate':
            checkpoint_path = config['checkpoint_dir'] + "/alphagenome_student_model.pkl"
            if not Path(checkpoint_path).exists():
                logger.error(f"Student model not found: {checkpoint_path}")
                return 1
            
            evaluate_model(config, checkpoint_path)
        
        elif args.stage == 'calibrate':
            checkpoint_path = config['checkpoint_dir'] + "/alphagenome_student_model.pkl"
            if not Path(checkpoint_path).exists():
                logger.error(f"Student model not found: {checkpoint_path}")
                return 1
            
            calibrate_variant_scorers(config, checkpoint_path)
        
        elif args.stage == 'full':
            logger.info("Running full training pipeline")
            
            # Run complete pipeline
            if args.demo:
                logger.info("Demo mode: skipping actual training")
                # Create dummy checkpoints for demo
                demo_checkpoint = Path(config['checkpoint_dir']) / "demo_model.pkl"
                demo_checkpoint.touch()
                logger.info(f"Demo checkpoint created: {demo_checkpoint}")
            else:
                results = train_full_pipeline(config)
                logger.info("Full pipeline completed successfully")
                logger.info(f"Results: {list(results.keys())}")
        
        logger.info("Training completed successfully!")
        return 0
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return 1


if __name__ == "__main__":
    exit(main()) 