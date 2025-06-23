
from video_predictor import predict_video_authenticity  # Import from the main script

def main():
    # Configuration
    MODEL_PATH = 'back_model.pth'  # Path to your trained model
    MASTER_CSV_PATH = '../master_features.csv'   # Path to your reference data
    TEST_CSV_PATH = 'test.csv'                # Path to your test video
    
    # Number of reference videos to compare with (adjust based on your resources)
    NUM_REFERENCE_VIDEOS = 100
    
    print("Starting video authenticity prediction...")
    print(f"Model: {MODEL_PATH}")
    print(f"Reference data: {MASTER_CSV_PATH}")
    print(f"Test video: {TEST_CSV_PATH}")
    print(f"Using {NUM_REFERENCE_VIDEOS} reference videos per class")
    print("-" * 50)
    
    try:
        # Run prediction
        results = predict_video_authenticity(
            model_path=MODEL_PATH,
            master_csv_path=MASTER_CSV_PATH,
            test_csv_path=TEST_CSV_PATH,
            num_reference_videos=NUM_REFERENCE_VIDEOS
        )
        
        
        # Quick summary
        print(f"\nQUICK SUMMARY:")
        print(f"Video is predicted to be: {results['prediction']['class']}")
        print(f"Confidence: {results['prediction']['confidence']:.4f}")
        
        return results
        
    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    results = main()