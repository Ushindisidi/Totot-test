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
Pistachio by Masala Twist

A refined fusion of Indian and Middle Eastern flavors.

---

**Menu Overview**

A taste of Pistachio.

**Main Course**
Our hearty mains are crafted with rich, authentic flavors—offering both vegetarian and non-vegetarian options, from classic paneer curries to grilled meats and biryanis.
Dish shown: Paneer Tikka Masala

**Starter**
Begin your journey with light, flavorful starters, from crispy vegetarian bites to spicy kebabs and Middle Eastern mezze.
Dish shown: Pesto Kebab

**Sides**
Complete your meal with fresh-baked breads, seasoned rice, and savory accompaniments to pair perfectly with every main.
Dish shown: Naan

**Dessert**
A sweet ending awaits with indulgent Indian and Middle Eastern treats: rich kulfi, delicate pastries, and fusion flavors you will remember.

---

**Make a Reservation**
Book your dinner experience in just a few clicks. A form is provided on the website, or you can use the contact details provided.

---

**Our Story**
At Pistachio by Masala Twist, food is more than a flavor—it’s a feeling.
Founded by the team behind the beloved Masala Twist, Pistachio offers a modern take on traditional Indian and Lebanese cuisine. Warm yet refined, our space is designed to celebrate culture, spice, and soul.
From slow-cooked curries to smoky grilled meats and fresh vegetarian thalis, our kitchen blends authenticity with a contemporary touch — crafted for those who appreciate thoughtful dining.

**Core Values:**
* Freshness: Ingredients sourced daily, prepared with care.
* Culture: Flavors inspired by heritage, served with heart.
* Experience: Every plate is part of a larger story.

---

**Featured Dishes**
(Pistachio’s most celebrated creations - curated for flavor, flair, and finesse)

* **Braised Mutton Champ Maple Kokum Glaze**
    Braised goat rack with maple and kokum syrup. Kokum plant, called 'garcinia indica', is grown in the Kokan regions of Maharashtra.

* **Chilli Chicken**
    Succulent boneless chicken tossed in spicy semi-dry sauce, garnished with spring onions.
    Ksh 1410

* **Prawns Kolapuri**
    A classic dish of prawns sautéed with in-house spicy Kolapuri spices.
    Ksh 580

* **Fish Chilli Fry**
    Red snapper marinated with Chinese spices, deep-fried, tossed with bell pepper, onions, and a dash of lemon.
    Ksh 1300

* **Chicken Biryani**
    Long grain basmati rice, succulent chicken, and a mélange of extra Indian spices cooked in a sealed deg that keeps the flavors intact.
    Ksh 780

* **Chicken Biryani**
    The most trending Indian dish across the globe which is proud of Indian cuisine! Boneless chicken cooked in tandoor, served with rich makhani gravy, dusted off with dehydrated fenugreek.
    Ksh 1050

* **Mac And Cheese Arancini**
    Macaroni and Cheese made in arancini dumpling, served with tomato salsa—our humble take on Italian arancini.
    Ksh 520

* **Chilli Garlic Button Mushrooms**
    Fresh button mushrooms tossed in sharp chili dry spices, topped with spring onions—a must-try for every mushroom lover.
    Ksh 320

---

**Reviews**
(What Our Guests Are Saying)

* “The butter chicken is absolutely phenomenal. Flavorful, rich, and unforgettable!”
    - Aisha K.
* “Loved the ambiance and warm service. Definitely coming back.”
    - David M.

---

**Feedback**
A feedback form is provided on the website.

---

**Contact Us**
14 Riverside Drive, Nairobi
+254 712 345678
info@pistachionairobi.com

---

**Opening Hours**
Mon-Fri: 11:00 AM - 10:00 PM
Sat-Sun: 12:00 PM - 11:00 PM

---

**Our Full Menu**
(Explore our Vibrant Indian and Middle Eastern Cuisine)

**Mains**
* **Hand Pulled Butter Chicken Makhani**
    The most trending Indian dish across the globe which is proud of Indian cuisine! Boneless chicken cooked in tandoor, served with rich makhani gravy, dusted off with dehydrated fenugreek.

* **Prawns Kolapuri**
    A classic dish of prawns sautéed with in-house spicy Kolapuri spices.

* **Hyderabadi Chicken**
    Succulent chicken pieces cooked in tandoor, simmered off with spices of Hyderabad.

* **Pan Toss Masala Shrooms Truffle Haze**
    Assorted mushrooms with cream garlic truffle oil and served with aromatic truffle oil haze.

* **Paneer Bhurji**
    Minced cottage cheese stir-fried with onions and tomato, lightly spiced with freshly pounded coriander and chilies.

* **Amritsari Malai Kofta**
    Cottage cheese dumplings, khoya, raisin, nuts, green chilies, cashew gravy, and can be served Jain on request.

**Starters**
* **Braised Mutton Champ Maple Kokum Glaze**
    Braised goat rack with maple and kokum syrup. Kokum plant, called 'garcinia indica', is grown in the Kokan regions of Maharashtra.

* **Chilli Chicken**
    Crispy fried chicken tossed in a fiery Indo-Chinese sauce made with garlic, soy, and chili, garnished with spring onions. A bold fusion favorite, perfect for spice lovers.

* **Chilli Garlic Button Mushrooms**
    Juicy button mushrooms stir-fried in a zesty garlic-chili sauce, finished with a hint of soy and a sprinkle of spring onions. A savory delight with a spicy kick.

* **Fish Chilli Fry**
    Red snapper marinated with Chinese spices, deep-fried, tossed with bell pepper, onions, and a dash of lemon.

* **Mac And Cheese Arancini**
    Macaroni and Cheese made in arancini dumpling, served with tomato salsa—our humble take on Italian arancini.

* **Pesto Kebab**
    Kebabs made of Italian pesto basil sauce, flavored with roasted walnuts, almonds, and cashew nuts, and can be served Jain on request.

**Sides**
* **3 Cheese Naan Basil Butter**
    Mozzarella, parmesan, and smoked cheddar melted into soft tandoor-baked naan, finished with a brush of aromatic basil butter—rich, gooey, and indulgent.

* **Naan**
    Fluffy, hand-stretched flatbread cooked in the tandoor, offering a smoky char and the perfect companion to hearty Indian gravies.

* **Jeera Rice**
    Fragrant basmati rice tempered with roasted cumin seeds and ghee—a comforting, flavorful side that complements any main.

* **Mushroom Corn Rice**
    A colorful medley of mushrooms, corn, and aromatic rice tossed with Indian spices—a flavorful, hearty vegetarian delight.

* **Plain Steam Rice**
    Steamed to soft, fluffy perfection, this simple staple lets the bold flavors of your curry shine through.

* **Cucumber and Mint Raita**
    Cool cucumber, crisp mint, and creamy yogurt blended into a refreshing side dish that balances spice with every bite.

**Desserts**
* **Assorted Ice Cream**
    A classic medley of rich, creamy ice cream flavors—the perfect sweet finish to any meal, served chilled and garnished with mint.

* **Gajar Halwa**
    A warm North Indian dessert made from slow-cooked grated carrots in ghee and milk, sweetened and topped with roasted cashews and almonds.

* **Indian Masala Chai Ice Cream**
    A bold twist on dessert—smooth ice cream infused with the warming spices of masala chai: cardamom, cinnamon, clove, and tea essence.

* **Gulab Jamun**
    Soft milk-solid dumplings soaked in fragrant rose and cardamom syrup—an indulgent, melt-in-my-mouth delight.

* **Malai Kulfi**
    Traditional Indian ice cream made with condensed milk and cream, infused with cardamom and pistachio, frozen on sticks for nostalgic flair.

---

**Pistachio Ice Cream** (Special Highlight)
Creamy and nutty with a luxurious texture, this pistachio-flavored treat offers subtle sweetness and a touch of crunch in every spoonful.
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
