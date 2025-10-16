import google.generativeai as genai

# Configure Gemini with your API key
GEMINI_API_KEY = "AIzaSyDvUYaiJzsH3CgYMogcS6BY3kPVzyV69hI"

if not GEMINI_API_KEY:
    print("Error: GEMINI_API_KEY not set.")
else:
    genai.configure(api_key=GEMINI_API_KEY)

    print("=" * 70)
    print("Available Gemini Models:")
    print("=" * 70)
    
    try:
        for m in genai.list_models():
            # Filter for models that can generate content (text generation)
            if 'generateContent' in m.supported_generation_methods:
                print(f"\n✓ {m.name}")
                print(f"  Description: {m.description}")
                print(f"  Version: {m.version}")
                print(f"  Methods: {', '.join(m.supported_generation_methods)}")
            else:
                print(f"\n- {m.name}")
                print(f"  Description: {m.description}")
                print(f"  Methods: {', '.join(m.supported_generation_methods)}")
        
        print("\n" + "=" * 70)
        print("Models with ✓ can be used for your chat feature.")
        print("=" * 70)

    except Exception as e:
        print(f"\nAn error occurred while listing models: {e}")
        print("Please check your API key and internet connection.")
        import traceback
        traceback.print_exc()