"""Snippet selection utilities for LKIF.

This module provides BM25 + rule-based snippet extraction
on top of QA-style datasets, to produce shorter references
(ground_truth.references) before feeding into the LKIF encoder
and training pipeline.
"""
