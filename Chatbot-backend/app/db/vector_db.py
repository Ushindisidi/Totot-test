import os
from langchain_openai import OpenAIEmbeddings # Import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS # Import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document # Import Document
from app.core.config import settings

# --- Data for the vector store (replace with your actual restaurant data) ---
# For a real application, you would load this from a file (e.g., a markdown file)
# or a database. For this example, we'll use a hardcoded string.
RESTAURANT_DATA = """
# Totot Traditional Food Hall (ቶቶት ገርጂ)

## 📍 Location & Contact
- *Address:* 2R54+2W4, Gerji, Bole Sub-city, Addis Ababa, Ethiopia
- *GPS Coordinates:* Approx. 9.0071° N, 38.8062° E
- *Phone:* +251 11 646 0718
- *Opening Hours:* 24 hours a day, 7 days a week
- *Nearest Landmark:* In front of World Vision HQ, near Anbesa Garage
- *Public Transport:* ~915m from Legehar Train Station
- *Parking:* Street parking only (no on-site lot)
- *Website:* http://www.totottraditionalrestaurant.com

---

## 🏛 Overview & Concept
- *Established:* Around 2001 E.C.
- *Name Origin:* “Totot” means “Let’s work” in the Gurage language.
- *Cuisine Focus:* Southern Ethiopian and Gurage cuisine.
- *Experience:* A full cultural immersion—food, drink, dance, music, coffee ceremony.

---

## 📋 Full Menu & Prices (ETB)

| Dish                     | Description                                                         | Price (ETB) |
|--------------------------|---------------------------------------------------------------------|-------------|
| *Traditional Special Lunch* | Mixed traditional meat dishes                                      | 480         |
| *Fasting Special*      | Vegan combination plate of national dishes                         | 350         |
| *Chikina Tibs*         | Sautéed beef fillet with onions                                    | 480         |
| *Zeilzil Tibs*         | Sliced beef with garlic and onions                                 | 480         |
| *Yebeg Tibs*           | Lamb sautéed with onions and spices                                | 480         |
| *Yebeg Wot*            | Traditional spicy lamb stew                                        | 450         |
| *Doro Wot*             | Ethiopian chicken stew with boiled egg                             | 450         |
| *Kitfo*                | Minced raw beef with spiced butter and mitmita                     | 480         |
| *Totot Kitfo*          | Special house-style version of Kitfo                               | 500         |
| *Tikur Kitfo*          | Premium Kitfo cooked in seasoned butter                            | 500         |
| *Gored Gored*          | Cubed raw beef with butter and spices                              | 500         |
| *Gomen Kitfo*          | Minced cabbage with spiced butter (vegan)                          | 450         |
| *Cow Tripa (Tripe)*    | Beef tripe stew with onions and carrots                            | 350         |
| *Gomen Besiga*         | Sautéed cabbage and meat in garlic and onion                       | 400         |
| *Bozena Shiro*         | Chickpea stew with meat chunks                                     | 320         |
| *Fintafinto*           | Cabbage cooked with minced beef in seasoned butter                 | 470         |
| *Kocho*                | False banana flatbread (side dish)                                 | 350         |
| *Zemamujet*            | Cottage cheese with local cabbage                                  | 350         |

> All prices include 15% VAT and service charge. Prices may vary by branch or during performances.

---

## 🍷 Drinks
- *Tej:* Traditional Ethiopian honey wine
- *Areke (Areqe):* Strong local liquor
- *Draft Beer & Local Brands*
- *Soft drinks & Water*
- *Coffee Ceremony:* Traditional Ethiopian coffee experience

Beer: ~37 ETB per bottle, Water: ~59 ETB

---

## 🧭 Directions
- *Google Maps:* Search Totot Traditional Food Hall Gerji
- *From Bole Airport:* 10–15 minutes by taxi via Bole–Gerji road.
- *Landmarks:* Opposite World Vision office, close to Anbesa Garage
- *Recommended Transport:* Ride-hailing apps like Ride or Feres.

---

## 🎶 Atmosphere & Entertainment
- *Decor:* Traditional huts, woven décor, low tables, cultural motifs
- *Live Shows:* Every night—tribal dances, music, and audience participation
- *Best Time to Visit:* Sunday evenings for full entertainment vibe
- *Coffee Ceremony:* Available upon request; includes roasting, serving

---

## ✅ Visitor Tips
- Arrive before 7:00 PM to avoid crowd surges
- Reserve a table for group visits or weekend nights
- Bring cash – cards may not always be accepted
- No on-site parking; street parking only
- Expect performances to raise noise levels in the evening

---

## ⭐ Ratings & Reviews
- *Google:* ~4.1/5 based on 360+ reviews
- *TripAdvisor:* Mixed reviews praising food & culture, some criticize service
- *Reddit Review:* “Has the best t’ej in town!”
- *Positive Notes:*
  - Best Kitfo in Addis
  - Generous meat portions
  - Authentic experience
- *Criticisms:*
  - Expensive by local standards
  - Occasional service delays or attitude issues
  - One reviewer claimed illness (isolated case)

---

## 🏆 Awards & Recognition
- 🥇 *Best Restaurant in Ethiopia – 2020*  
  World Culinary Awards

---

## 🧾 Summary

```text
Name:           Totot Traditional Food Hall (Totot Kitfo)
Location:       Gerji, Bole Sub-city, Addis Ababa, Ethiopia
Phone:          +251 11 646 0718
Hours:          Open 24/7
Cuisine:        Ethiopian (Gurage, Kitfo, Tibs, Wot)
Menu Price:     320–500 ETB
Drinks:         Tej, Areqe, Beer, Coffee
Specials:       Live dance, music, coffee ceremony
Rating:         ~4.1/5 based on 360 reviews
Reservations:   Recommended for weekends
"""

# Initialize OpenAI Embeddings
# model_kwargs={"device": "cpu"} is not needed for OpenAI API
embeddings = OpenAIEmbeddings(
    openai_api_key=settings.OPENAI_API_KEY,
    model="text-embedding-ada-002" # Or a newer embedding model if available
)

def get_faiss_vectorstore():
    """
    Creates and returns an in-memory FAISS vector store from the restaurant data.
    """
    try:
        if not settings.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY is not set in environment variables. Cannot create embeddings.")

        # Split the document into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            is_separator_regex=False,
        )
        texts = text_splitter.split_text(RESTAURANT_DATA)

        # Convert text chunks to Document objects
        documents = [Document(page_content=t) for t in texts]

        # Create FAISS vector store from documents and embeddings
        docsearch = FAISS.from_documents(documents, embeddings)
        print("Successfully created in-memory FAISS vector store with OpenAI Embeddings.")
        return docsearch
    except Exception as e:
        print(f"Error creating FAISS vector store: {e}")
        raise # Re-raise the exception to be caught by FastAPI's error handling

# Initialize the vector store at startup
docsearch = None
try:
    docsearch = get_faiss_vectorstore()
except Exception as e:
    print(f"Failed to initialize FAISS vector store at startup: {e}")
    # docsearch remains None, and the chat endpoint will handle this.
