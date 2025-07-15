



# ==============================================================================
# STEP 1: INSTALL LIBRARIES AND CONFIGURE API
# ==============================================================================
import streamlit as st
import google.generativeai as genai
import pandas as pd
import numpy as np
import json
import re

# ==============================================================================
# STEP 2: LOAD & PREPARE DATASETS WITH CACHING
# ==============================================================================

# --- Page and Model Configuration ---
st.set_page_config(page_title="Autovisory - AI Car Analyst", page_icon="🚗", layout="wide")

# --- Load API Key and Configure AI Model ---
try:
    GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
    genai.configure(api_key=GOOGLE_API_KEY)
    model = genai.GenerativeModel('models/gemini-2.5-pro')
except Exception as e:
    st.error(f"Error loading API key or configuring the model: {e}")
    st.stop()



@st.cache_data
def load_and_prepare_data():
    """Loads all datasets and prepares them for the app."""
    try:
        # Load data from GitHub Raw URLs
        url_gas = "https://raw.githubusercontent.com/Belayet-Sheikh/Autovisory-AI-Carbot/main/Data/data.csv"
        url_ev = "https://raw.githubusercontent.com/Belayet-Sheikh/Autovisory-AI-Carbot/main/Data/electric-vehicle-population-data.csv"
        url_used_us = "https://raw.githubusercontent.com/Belayet-Sheikh/Autovisory-AI-Carbot/main/Data/vehicles.csv"
        url_used_europe = "https://raw.githubusercontent.com/Belayet-Sheikh/Autovisory-AI-Carbot/main/Data/car_price.csv"

        df_gas = pd.read_csv(url_gas)
        df_ev = pd.read_csv(url_ev)
        df_used_us = pd.read_csv(url_used_us)
        df_used_europe = pd.read_csv(url_used_europe)

        # === FINAL CORRECTED LOGIC ===
        
        # --- Process Gas Cars ---
        df_gas.columns = df_gas.columns.str.replace(' ', '_').str.lower()
        df_gas = df_gas.rename(columns={'msrp': 'price'})
        df_gas['fuel_type'] = 'Gasoline'
        # Ensure the 'electric_range' column exists, filling with 0
        df_gas['electric_range'] = 0

        # --- Process EV Cars ---
        df_ev.columns = df_ev.columns.str.replace(' ', '_').str.lower()
        df_ev = df_ev.rename(columns={'model_year': 'year'})
        df_ev['fuel_type'] = 'Electric'
        # IMPORTANT: Manually create the columns that are missing in the EV dataset
        df_ev['engine_hp'] = np.nan
        df_ev['city_mpg'] = np.nan
        df_ev['vehicle_style'] = 'N/A' # Use a placeholder like 'N/A' or np.nan

        # --- Create a single list of columns that both datasets will now have ---
        cols_to_keep = ['make', 'model', 'year', 'price', 'vehicle_style', 'engine_hp', 'city_mpg', 'fuel_type', 'electric_range']
        
        # Select only these columns from each dataframe. This is now safe.
        df_gas_processed = df_gas[cols_to_keep]
        df_ev_processed = df_ev[cols_to_keep]

        # Concatenate the processed dataframes
        df_new_us_master = pd.concat([df_gas_processed, df_ev_processed], ignore_index=True)
        
        # === END OF FINAL CORRECTION ===
        
        df_new_us_master = df_new_us_master.dropna(subset=['year', 'make', 'model'])
        df_new_us_master['year'] = df_new_us_master['year'].astype(int)

        df_used_us = df_used_us.rename(columns={'manufacturer': 'make'})
        used_us_cols = ['make', 'model', 'year', 'price', 'odometer']
        df_used_us_master = df_used_us[used_us_cols].dropna()
        df_used_us_master = df_used_us_master[df_used_us_master['price'].between(100, 250000)]
        df_used_us_master['year'] = df_used_us_master['year'].astype(int)
        df_used_us_master['odometer'] = df_used_us_master['odometer'].astype(int)

        df_used_europe = df_used_europe.rename(columns={'Brand': 'make', 'Model': 'model', 'Year': 'year', 'Price': 'price', 'Kilometers': 'odometer'})
        used_europe_cols = ['make', 'model', 'year', 'price', 'odometer']
        df_used_europe_master = df_used_europe[used_europe_cols].dropna()
        df_used_europe_master['year'] = pd.to_numeric(df_used_europe_master['year'], errors='coerce').dropna().astype(int)
        df_used_europe_master['odometer'] = pd.to_numeric(df_used_europe_master['odometer'], errors='coerce').dropna().astype(int)

        return df_new_us_master, df_used_us_master, df_used_europe_master

    except Exception as e:
        st.error(f"An error occurred while loading or processing data: {e}")
        st.stop()


# Load the data
df_new_us_master, df_used_us_master, df_used_europe_master = load_and_prepare_data()


# ==============================================================================
# STEP 3: AI INTENT & RESPONSE LOGIC
# ==============================================================================

def determine_next_action(history, user_query):
    history_str = "\n".join([f"{h['role']}: {h['parts']}" for h in history])
    prompt = f"""
    You are Autovisory, a helpful, expert, and impartial AI assistant specializing exclusively in cars. Your primary mission is to be a trusted advisor, guiding users through the complexities of buying and understanding cars. You are not a salesperson; your advice is objective and always centered on the user's needs.


    Step 1: Initial Query Assessment
First, analyze the user's query.
If the query is about anything NOT related to cars (e.g., movies, weather, politics, recipes), you MUST respond only with the following JSON object and nothing else:
{{"action": "reject", "response": "I'm here to help only with car-related questions. Could you ask something about cars?"}}
If the query is car-related, proceed to Step 2 to determine the user's intent.

    Step 2: Intent-Based Action Protocol
Decide the user's intent and follow the corresponding protocol precisely.
Intent: clarify
Condition: The user's query is vague, unclear, or they state they don't know where to start (e.g., "What car should I buy?", "I need a new car").
Action: Your goal is to understand their needs. Ask a series of clarifying questions to gather essential information. Do not suggest any cars yet. Your response should be a friendly set of questions.
Questions to Ask:
"To give you the best recommendation, I need a little more information. Could you tell me about:"
"1. Budget: What is your approximate budget for the car?"
"2. Primary Use: What will you mainly be using it for (like daily commuting, family trips, or off-roading)?"
"3. Passengers: How many people will you typically need to carry?"
"4. Priorities: What are the top 3 most important things for you in a car (e.g., fuel efficiency, safety, performance, reliability, cargo space, latest tech)?"
"5. Lifestyle: Do you have any specific needs, like carrying pets, sports gear, or needing to tow anything?"
Intent: recommend
Condition: The user has provided enough information for a recommendation, either in their initial query or after you've clarified.
Action: Suggest 2-3 well-suited car models. For each suggestion, provide a brief, compelling summary explaining why it fits their stated needs. Mention key strengths related to their priorities.
Example Output Structure: "Based on your need for a fuel-efficient family car under $30,000, here are a couple of great options:
Honda CR-V: It's known for its outstanding reliability and has a huge, practical interior, making it perfect for family duties. Its fuel economy is also excellent for its class.
Toyota RAV4 Hybrid: This would be a top choice for maximizing fuel efficiency. It also comes standard with many safety features and has a strong reputation for holding its value."
Intent: analyze
Condition: The user wants information on a specific car model (e.g., "Tell me about the 2024 Ford Mustang," "What do you know about the Kia Telluride?").
Action: Provide a comprehensive yet easy-to-digest overview of the vehicle.
Information to Include:
Summary: A brief paragraph on what the car is and who it's for.
Pros: List 3-4 key strengths (e.g., "Powerful engine options," "High-quality interior," "Excellent safety scores").
Cons: List 3-4 key weaknesses (e.g., "Poor rear visibility," "Below-average fuel economy for its class," "Stiff ride").
Key Specifications: Mention engine choices, horsepower, fuel economy (MPG/L/100km), and drivetrain options.
Safety & Reliability: Reference ratings from trusted sources like the IIHS or NHTSA if available.
Intent: compare
Condition: The user wants to compare two or more car models (e.g., "Honda Civic vs. Toyota Corolla," "Compare the F-150 and the Silverado").
Action: Create a structured, direct comparison. A side-by-side table is highly effective. After the table, provide a concluding summary.
Comparison Points: Always include Price Range, Fuel Economy, Performance (engine/HP), Interior/Cargo Space, and Safety/Reliability Ratings. Add other points as relevant (e.g., Towing Capacity for trucks, EV Range for electric cars).
Concluding Summary: Briefly state which car is better for different types of buyers. (e.g., "The Civic may be better for those who prioritize a fun driving experience, while the Corolla is a top choice for those focused on maximum reliability and comfort.")
Intent: answer_general
Condition: The user has a general knowledge question about cars, brands, or technology (e.g., "What's the difference between a hybrid and a PHEV?", "Are Kia cars reliable?", "What is a CVT?").
Action: Provide a clear, accurate, and simple explanation. Avoid overly technical jargon. Your goal is to educate the user.

Step 3: Overarching Guiding Principles (Apply to all car-related responses)
Impartiality: Never show bias towards a brand. Always present a balanced view.
Safety First: Always prioritize safety. Highlight vehicles with strong safety ratings.
Data-Driven: When citing safety or reliability, you can mention the source (e.g., "According to the IIHS...").
No Financial Advice: You can discuss a car's price (MSRP), but do not advise a user on what they can afford or how to finance a vehicle.
Honesty: If you don't have specific information, state that it's unavailable rather than guessing.
If the use say thank you or thank you for help. Then reply with welcome and ask they need more help of not.

    USER QUERY: "{user_query}"
    Conversation History:\n{history_str}
    Return only valid JSON.
    """
    try:
        response = model.generate_content(prompt)
        text = response.text.strip().replace('```json', '').replace('```', '')
        return json.loads(text)
    except Exception as e:
        st.error(f"An error occurred while determining the action: {e}")
        return {"action": "error", "response": "Sorry, I had trouble understanding that. Could you rephrase?"}

def extract_car_models(text):
    pattern = r'(?:\b(?:about|buy|vs|compare|between|more on|tell me about|interested in|details on)\b)?[\s:,-]*([A-Z][a-z]+(?: [A-Z0-9][a-z0-9]*){0,3})'
    return re.findall(pattern, text, flags=re.IGNORECASE)


def get_recommendations_and_analysis(full_context_query):
    prompt = f"""
    You're an expert AI Car Analyst. Recommend 3 cars based on the context and analyze them.

    CONTEXT:
    {full_context_query}

    Datasets:
    NEW US CARS (sampled), USED US CARS (sampled), USED EUROPEAN CARS (sampled)

    1. Pick 3 cars matching budget & type (new/used).
    2. For each, provide:
      - Summary
      - US New Price
      - Avg US Used Price (within budget)
      - Avg EU Used Price (within budget)
    3. Respond in structured JSON as:
    {{
      "recommendations": [
        {{
          "make": "Toyota",
          "model": "Camry",
          "summary": "Reliable, fuel-efficient midsize sedan...",
          "us_market": {{"new_car_price_usd": 24000, "average_used_price_usd": 19500}},
          "europe_market": {{"average_used_price_eur": 17000}}
        }},
        ...
      ]
    }}
    """
    try:
        # Note: In a real app, you'd pass these dataframes to the model.
        # For now, we simulate this by having the data loaded in the app's context.
        response = model.generate_content(prompt)
        text = response.text.strip().replace('```json', '').replace('```', '')
        return json.loads(text)
    except Exception as e:
        return {"error": str(e)}

def compare_cars_with_ai(full_context_query):
    prompt = f"""
    You are a car expert AI. The user is trying to decide between two or more vehicles.

    Based on this conversation:
    {full_context_query}

    Compare the models side-by-side in this structured JSON:
    {{
      "comparison": [
        {{
          "brand": "Tesla",
          "summary": "Known for cutting-edge EVs with long range and advanced technology.",
          "strengths": ["Long battery range", "Fast charging (Supercharger)", "Strong resale value"],
          "weaknesses": ["Higher upfront cost", "Minimalist design may not appeal to everyone"]
        }},
        {{
          "brand": "Ford",
          "summary": "Offers practical EVs like the Mustang Mach-E and F-150 Lightning, blending traditional design with new tech.",
          "strengths": ["Lower price entry", "Familiar interior", "Solid ride quality"],
          "weaknesses": ["Fewer charging stations", "Range slightly lower than Tesla"]
        }}
      ]
    }}
    """
    try:
        response = model.generate_content(prompt)
        clean = response.text.strip().replace("```json", "").replace("```", "")
        return json.loads(clean)
    except Exception as e:
        return {"error": str(e), "comparison": []}

def analyze_specific_car_model(car_model):
    prompt = f"""
    You are an expert automotive analyst. Give a clear, concise analysis of the following car model:

    Model: "{car_model}"

    Your response must include:
    - An overview of the vehicle
    - Common pros and cons
    - Target audience
    - Typical market price range (if known)

    Structure the output as JSON like this:
    {{
      "model": "Tesla Model Y",
      "overview": "...",
      "pros": [...],
      "cons": [...],
      "audience": "...",
      "price_estimate_usd": "..."
    }}
    """
    try:
        response = model.generate_content(prompt)
        text = response.text.strip().replace('```json', '').replace('```', '')
        return json.loads(text)
    except Exception as e:
        return {"error": str(e)}

# ==============================================================================
# STEP 4: STREAMLIT CHAT INTERFACE
# ==============================================================================

st.title("🚗 Autovisory: AI Car Market Analyst")
st.write("Your expert guide to the global car market. Ask me anything about cars!")

# Initialize chat history in session state
if "history" not in st.session_state:
    st.session_state.history = []

# Display chat messages from history
for message in st.session_state.history:
    with st.chat_message(message["role"]):
        st.markdown(message["parts"])

# Accept user input
if user_prompt := st.chat_input("What car are you looking for?"):
    # Add user message to history and display it
    st.session_state.history.append({"role": "user", "parts": user_prompt})
    with st.chat_message("user"):
        st.markdown(user_prompt)

    # Get AI response
    with st.chat_message("assistant"):
        with st.spinner("Autovisory is thinking..."):
            action_data = determine_next_action(st.session_state.history, user_prompt)
            action = action_data.get("action")
            response_text = ""

            if action == "reject":
                response_text = action_data.get("response", "I can only answer car-related questions.")
                st.markdown(response_text)

            elif action == "clarify":
                response_text = action_data.get("response", "I need more details to help you. What's your budget and primary use case?")
                st.markdown(response_text)

            elif action == "answer_general":
                 response_text = action_data.get("response", "That's a great question! Let me explain...")
                 st.markdown(response_text)

            elif action == "recommend":
                full_context = "\n".join([f"{msg['role']}: {msg['parts']}" for msg in st.session_state.history])
                recs = get_recommendations_and_analysis(full_context)
                if recs.get("recommendations"):
                    response_text = "Based on your preferences, here are 3 solid options:"
                    st.markdown(response_text)
                    for r in recs["recommendations"]:
                        st.markdown(f"#### 🚗 **{r['make']} {r['model']}**")
                        st.markdown(f"**Summary**: {r['summary']}")
                        us_market = r.get('us_market', {})
                        eu_market = r.get('europe_market', {})
                        st.markdown(f"**Prices:**")
                        col1, col2, col3 = st.columns(3)
                        col1.metric("New (US)", f"${us_market.get('new_car_price_usd', 'N/A'):,}", delta_color="off")
                        col2.metric("Used (US)", f"${us_market.get('average_used_price_usd', 'N/A'):,}", delta_color="off")
                        col3.metric("Used (EU)", f"€{eu_market.get('average_used_price_eur', 'N/A'):,}", delta_color="off")
                else:
                    response_text = "Sorry, I couldn't find good options with the provided details. Could you try rephrasing?"
                    st.markdown(response_text)

            elif action == "analyze":
                candidates = extract_car_models(user_prompt)
                model_name = candidates[0] if candidates else ""
                analysis = analyze_specific_car_model(model_name)
                if analysis.get("model"):
                    response_text = f"Here's a detailed analysis of the **{analysis['model']}**:"
                    st.markdown(response_text)
                    st.markdown(f"**📘 Overview:** {analysis['overview']}")
                    st.markdown(f"**✅ Pros:** {', '.join(analysis['pros'])}")
                    st.markdown(f"**⚠️ Cons:** {', '.join(analysis['cons'])}")
                    st.markdown(f"**👥 Ideal For:** {analysis['audience']}")
                    st.metric("Estimated Price", analysis['price_estimate_usd'])
                else:
                    response_text = "Sorry, I couldn't find detailed information for that specific model."
                    st.markdown(response_text)

            elif action == "compare":
                 full_context = "\n".join([f"{msg['role']}: {msg['parts']}" for msg in st.session_state.history])
                 result = compare_cars_with_ai(full_context)
                 if result.get("comparison"):
                     response_text = "Here's a comparison of your choices:"
                     st.markdown(response_text)
                     for car in result["comparison"]:
                         with st.expander(f"🚘 **{car['brand']}**"):
                             st.markdown(f"**📝 Summary:** {car['summary']}")
                             st.markdown(f"**✅ Strengths:**\n" + "\n".join([f"- {s}" for s in car['strengths']]))
                             st.markdown(f"**⚠️ Weaknesses:**\n" + "\n".join([f"- {w}" for w in car['weaknesses']]))
                 else:
                     response_text = "I couldn't generate a comparison. Please make sure you mention at least two cars."
                     st.markdown(response_text)

            else:
                response_text = action_data.get("response", "I'm not sure how to respond to that. Please try another question.")
                st.markdown(response_text)

            # Add AI response to history
            st.session_state.history.append({"role": "assistant", "parts": response_text})
