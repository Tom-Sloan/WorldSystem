# StreamingSLAM3R Architecture Comparison

## Why the New Architecture is Better

### Old Approach Issues

1. **Mixed State Management**
   - Tokens stored in different formats at different times
   - Conditional dimension handling throughout the code
   - Complex history tracking with multiple representations

2. **Dimension Confusion**
   - Constant `unsqueeze(0)` and `squeeze(0)` operations
   - Uncertainty about expected tensor shapes
   - Model expecting batch processing, getting streaming data

3. **Tight Coupling**
   - Processing logic mixed with RabbitMQ handling
   - Visualization code intertwined with SLAM logic
   - No clear separation of concerns

### New Architecture Benefits

1. **Clean Abstractions**
   ```python
   # Old: Mixed concerns
   if tok[0].dim() == 4:
       tok = tok[0].squeeze(0)
   record["img_tokens"] = tok
   
   # New: Clear responsibility
   frame.img_tokens = self.token_cache.normalize_and_store(tokens)
   ```

2. **Consistent Data Flow**
   - `FrameData` class ensures consistent structure
   - `ViewFactory` creates properly formatted views
   - No conditional dimension handling in main logic

3. **Batch Processing Support**
   - `BatchAccumulator` handles frame batching transparently
   - Works WITH model expectations, not against them
   - Better GPU utilization

4. **State Management**
   - `TokenCache` manages all token storage
   - `SlidingWindowProcessor` handles reference frames
   - Clear ownership and lifecycle

5. **Easier Debugging**
   - Each component has single responsibility
   - Clear data flow through pipeline
   - Predictable tensor shapes at each stage

## Key Design Patterns

### 1. Factory Pattern
```python
# Ensures consistent view creation
view = ViewFactory.create_view(frame, include_image=True)
```

### 2. Adapter Pattern
```python
# Handles impedance mismatch
prepared_views = ViewFactory.prepare_for_model(views, add_batch_dim=True)
```

### 3. Repository Pattern
```python
# Clean token storage
self.token_cache.add(frame_id, tokens)
tokens = self.token_cache.get(frame_id)
```

### 4. Pipeline Pattern
```python
# Clear processing stages
frame → generate_tokens → accumulate → process_batch → publish
```

## Performance Improvements

1. **Batching**: Process multiple frames together for better GPU utilization
2. **Caching**: Avoid regenerating tokens for reference frames
3. **Memory Management**: Automatic cleanup of old tokens
4. **Async Processing**: Non-blocking frame handling

## Migration Path

1. **Phase 1**: Deploy new architecture alongside old one
2. **Phase 2**: Route subset of traffic to new processor
3. **Phase 3**: Monitor and compare results
4. **Phase 4**: Gradually increase traffic to new processor
5. **Phase 5**: Deprecate old processor

## Configuration

The new architecture is highly configurable:

```bash
# Environment variables
SLAM3R_BATCH_SIZE=5          # Frames per batch
SLAM3R_WINDOW_SIZE=20        # Sliding window size
SLAM3R_INIT_KF_STRIDE=5      # Initial keyframe stride
SLAM3R_INIT_FRAMES=5         # Frames for initialization
SLAM3R_CONF_THRES=5.0        # Confidence threshold
```

## Extensibility

Adding new features is straightforward:

```python
# Add new processing stage
class FeatureExtractor:
    def extract(self, frame: FrameData):
        # New feature extraction logic
        pass

# Integrate into pipeline
self.feature_extractor = FeatureExtractor()
features = self.feature_extractor.extract(frame)
```

## Conclusion

The new StreamingSLAM3R architecture provides:
- Clean separation of concerns
- Predictable behavior
- Better performance
- Easier maintenance
- Clear extension points

It works WITH the batch-oriented SLAM3R models rather than fighting against them, resulting in a more robust and maintainable system.