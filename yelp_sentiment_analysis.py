import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import re
import warnings
warnings.filterwarnings('ignore')

from textblob import TextBlob
import nltk
nltk.download('stopwords', quiet=True)
nltk.download('punkt',     quiet=True)
from nltk.corpus import stopwords

def generate_yelp_data():
    np.random.seed(42)
    restaurants = {
        'The Golden Fork':   'Italian',
        'Spice Garden':      'Indian',
        'Burger Republic':   'American',
        'Tokyo Ramen House': 'Japanese',
        'La Bella Napoli':   'Italian',
        'Dragon Palace':     'Chinese',
        'Taco Fiesta':       'Mexican',
        'The Steakhouse':    'American',
        'Sushi Paradise':    'Japanese',
        'Curry Kingdom':     'Indian',
    }
    positive_reviews = [
        "Absolutely amazing food, best restaurant in the city!",
        "Incredible flavors, friendly staff and cozy atmosphere.",
        "Loved every bite, will definitely come back again soon!",
        "Outstanding service and the pasta was perfectly cooked.",
        "Fantastic experience from start to finish, highly recommend!",
        "The food was divine, fresh ingredients and generous portions.",
        "Exceptional quality and the dessert was absolutely heavenly.",
        "Perfect date night spot, romantic ambiance and delicious food.",
        "Best meal I have had in years, exceeded all expectations!",
        "Wonderful staff, quick service and incredibly tasty dishes.",
        "Hidden gem in the city, every dish was a masterpiece.",
        "Great value for money and the flavors were authentic.",
        "Five stars all the way, food was piping hot and delicious!",
        "Loved the ambiance and the chef really knows their craft.",
        "Superb dining experience, will bring my family next time.",
    ]
    neutral_reviews = [
        "Decent food but nothing that blew me away really.",
        "Average experience, food was okay and service was fine.",
        "It was alright, not the best but certainly not the worst.",
        "Pretty standard restaurant, food came out on time.",
        "Nothing special but would probably return if nearby.",
        "Met my expectations, no more and no less honestly.",
        "Food was okay, a bit pricey for what you get though.",
        "Reasonable place, the pasta was decent but uninspiring.",
        "Neither impressed nor disappointed, just an average meal.",
        "Standard fare, nothing memorable about the experience.",
    ]
    negative_reviews = [
        "Terrible service, waited over an hour for cold food!",
        "Worst restaurant experience I have ever had, never returning.",
        "Overpriced and the food was bland and disappointing.",
        "Rude staff and the portion sizes were ridiculously small.",
        "Found a hair in my food and the waiter was dismissive.",
        "Stale ingredients and the whole place smelled unpleasant.",
        "Absolutely disgusting, sent food back twice and left hungry.",
        "Complete waste of money, microwave quality at restaurant prices.",
        "Horrible experience, the manager was unhelpful and rude.",
        "Food took forever and arrived cold, totally unacceptable.",
        "Never again! Dry overcooked meat and soggy vegetables.",
        "Appalling hygiene standards and tasteless food throughout.",
    ]
    rows = []
    for restaurant, cuisine in restaurants.items():
        n = np.random.randint(50, 80)
        for _ in range(n):
            stars = np.random.choice([1,2,3,4,5], p=[0.08,0.10,0.12,0.30,0.40])
            if stars >= 4:
                review = np.random.choice(positive_reviews)
                suffix = np.random.choice([
                    " The chef deserves a raise!",
                    " Perfect for special occasions.",
                    " Service was top notch.",
                    " Prices are very reasonable.",
                    ""
                ])
            elif stars == 3:
                review = np.random.choice(neutral_reviews)
                suffix = np.random.choice([
                    " Might try again someday.",
                    " Has potential to improve.",
                    " Service was acceptable.",
                    ""
                ])
            else:
                review = np.random.choice(negative_reviews)
                suffix = np.random.choice([
                    " Reported to health department.",
                    " Asked for a refund.",
                    " Will not be returning.",
                    ""
                ])
            rows.append({
                'restaurant':  restaurant,
                'cuisine':     cuisine,
                'review':      review + suffix,
                'stars':       stars,
                'useful_votes': np.random.randint(0, 50),
                'funny_votes':  np.random.randint(0, 20),
                'cool_votes':   np.random.randint(0, 30),
            })
    return pd.DataFrame(rows)

print("=" * 60)
print("SENTIMENT ANALYSIS: YELP RESTAURANT REVIEWS")
print("=" * 60)

df = generate_yelp_data()
df.to_csv("yelp_reviews.csv", index=False)
print(f"Dataset ready: {len(df)} reviews saved as 'yelp_reviews.csv'")

STOP_WORDS = set(stopwords.words('english'))

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'<.*?>', ' ', text)
    text = re.sub(r'[^a-z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def remove_stopwords(text):
    return ' '.join(w for w in text.split() if w not in STOP_WORDS and len(w) > 2)

def get_sentiment(text):
    analysis = TextBlob(text)
    polarity     = analysis.sentiment.polarity
    subjectivity = analysis.sentiment.subjectivity
    if polarity > 0.1:
        label = 'Positive'
    elif polarity < -0.1:
        label = 'Negative'
    else:
        label = 'Neutral'
    return pd.Series([polarity, subjectivity, label])

print("\nCleaning and analyzing reviews...")
df['clean_text']    = df['review'].apply(clean_text)
df['clean_no_stop'] = df['clean_text'].apply(remove_stopwords)
df[['polarity','subjectivity','sentiment']] = df['clean_text'].apply(get_sentiment)

print("\nSentiment Distribution:")
print(df['sentiment'].value_counts())
print("\nAverage Polarity by Sentiment:")
print(df.groupby('sentiment')['polarity'].mean().round(3))
print("\nAverage Star Rating by Sentiment:")
print(df.groupby('sentiment')['stars'].mean().round(2))
print("\nSentiment by Restaurant:")
print(df.groupby(['restaurant','sentiment']).size().unstack(fill_value=0))
print("\nAverage Polarity by Cuisine Type:")
print(df.groupby('cuisine')['polarity'].mean().round(3).sort_values(ascending=False))

pos_words = ' '.join(df[df['sentiment']=='Positive']['clean_no_stop']).split()
neg_words = ' '.join(df[df['sentiment']=='Negative']['clean_no_stop']).split()
top_pos   = Counter(pos_words).most_common(15)
top_neg   = Counter(neg_words).most_common(15)

DARK   = '#1a1a2e'
GREY   = '#16213e'
GREEN  = '#2dc653'
RED    = '#e63946'
YELLOW = '#ffd700'
BLUE   = '#4361ee'
WHITE  = '#f0f0f0'
COLORS = {'Positive': GREEN, 'Neutral': YELLOW, 'Negative': RED}

sns.set_theme(style='dark')
fig, axes = plt.subplots(3, 3, figsize=(20, 18), facecolor=DARK)
fig.suptitle('Yelp Restaurant Reviews — Sentiment Analysis',
             fontsize=22, fontweight='bold', color=WHITE, y=1.01)

def style(ax, title):
    ax.set_facecolor(GREY)
    ax.set_title(title, color=WHITE, fontsize=12, fontweight='bold', pad=8)
    ax.tick_params(colors=WHITE)
    ax.xaxis.label.set_color(WHITE)
    ax.yaxis.label.set_color(WHITE)
    for spine in ax.spines.values():
        spine.set_edgecolor('#2a2a4a')

sent_counts = df['sentiment'].value_counts()
bars = axes[0, 0].bar(sent_counts.index, sent_counts.values,
                      color=[COLORS[s] for s in sent_counts.index],
                      edgecolor=DARK, linewidth=1.2)
style(axes[0, 0], 'Overall Sentiment Distribution')
axes[0, 0].set_ylabel('Number of Reviews', color=WHITE)
for bar in bars:
    axes[0, 0].text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 1,
                    str(int(bar.get_height())),
                    ha='center', color=WHITE, fontweight='bold')

for sentiment, color in COLORS.items():
    data = df[df['sentiment'] == sentiment]['polarity']
    axes[0, 1].hist(data, bins=25, alpha=0.7, color=color, label=sentiment)
style(axes[0, 1], 'Polarity Score Distribution by Sentiment')
axes[0, 1].set_xlabel('Polarity (-1=Negative, +1=Positive)', color=WHITE)
axes[0, 1].set_ylabel('Count', color=WHITE)
axes[0, 1].legend(facecolor=GREY, labelcolor=WHITE)

for sentiment, color in COLORS.items():
    grp = df[df['sentiment'] == sentiment]
    axes[0, 2].scatter(grp['polarity'], grp['subjectivity'],
                       c=color, label=sentiment, alpha=0.4, s=25)
style(axes[0, 2], 'Polarity vs Subjectivity')
axes[0, 2].set_xlabel('Polarity', color=WHITE)
axes[0, 2].set_ylabel('Subjectivity', color=WHITE)
axes[0, 2].axvline(0, color='white', linestyle='--', linewidth=0.8, alpha=0.5)
axes[0, 2].legend(facecolor=GREY, labelcolor=WHITE)

star_polarity = df.groupby('stars')['polarity'].mean()
bar_colors_s  = [RED, RED, YELLOW, GREEN, GREEN]
axes[1, 0].bar(star_polarity.index.astype(str), star_polarity.values,
               color=bar_colors_s, edgecolor=DARK)
style(axes[1, 0], 'Average Polarity by Star Rating')
axes[1, 0].set_xlabel('Star Rating', color=WHITE)
axes[1, 0].set_ylabel('Mean Polarity', color=WHITE)
axes[1, 0].axhline(0, color='white', linestyle='--', linewidth=0.8, alpha=0.5)
for bar in axes[1, 0].patches:
    axes[1, 0].text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 0.005,
                    f'{bar.get_height():.3f}',
                    ha='center', color=WHITE, fontsize=8)

pos_labels, pos_vals = zip(*top_pos) if top_pos else ([''], [0])
axes[1, 1].barh(pos_labels[::-1], pos_vals[::-1], color=GREEN, edgecolor=DARK)
style(axes[1, 1], 'Top 15 Words — Positive Reviews')
axes[1, 1].set_xlabel('Frequency', color=WHITE)

neg_labels, neg_vals = zip(*top_neg) if top_neg else ([''], [0])
axes[1, 2].barh(neg_labels[::-1], neg_vals[::-1], color=RED, edgecolor=DARK)
style(axes[1, 2], 'Top 15 Words — Negative Reviews')
axes[1, 2].set_xlabel('Frequency', color=WHITE)

rest_sentiment = df.groupby(['restaurant','sentiment']).size().unstack(fill_value=0)
rest_sentiment = rest_sentiment[['Positive','Neutral','Negative']]
rest_sentiment.plot(kind='bar', ax=axes[2, 0], stacked=True,
                    color=[GREEN, YELLOW, RED], edgecolor=DARK, linewidth=0.5)
style(axes[2, 0], 'Sentiment Breakdown by Restaurant')
axes[2, 0].set_xlabel('Restaurant', color=WHITE)
axes[2, 0].set_ylabel('Number of Reviews', color=WHITE)
axes[2, 0].tick_params(axis='x', rotation=35)
axes[2, 0].legend(facecolor=GREY, labelcolor=WHITE, fontsize=8)

cuisine_polarity = df.groupby('cuisine')['polarity'].mean().sort_values(ascending=False)
colors_c = [GREEN if v > 0.2 else YELLOW if v > 0 else RED for v in cuisine_polarity.values]
axes[2, 1].bar(cuisine_polarity.index, cuisine_polarity.values,
               color=colors_c, edgecolor=DARK)
style(axes[2, 1], 'Average Sentiment Polarity by Cuisine Type')
axes[2, 1].set_ylabel('Mean Polarity', color=WHITE)
axes[2, 1].tick_params(axis='x', rotation=30)
axes[2, 1].axhline(0, color='white', linestyle='--', linewidth=0.8, alpha=0.5)

star_dist = df['stars'].value_counts().sort_index()
axes[2, 2].bar(star_dist.index.astype(str), star_dist.values,
               color=[RED, RED, YELLOW, GREEN, GREEN],
               edgecolor=DARK)
style(axes[2, 2], 'Star Rating Distribution')
axes[2, 2].set_xlabel('Stars', color=WHITE)
axes[2, 2].set_ylabel('Number of Reviews', color=WHITE)
for bar in axes[2, 2].patches:
    axes[2, 2].text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 0.5,
                    str(int(bar.get_height())),
                    ha='center', color=WHITE, fontsize=9)

plt.tight_layout()
plt.savefig('yelp_sentiment_analysis.png', dpi=150,
            bbox_inches='tight', facecolor=DARK)
plt.show()
print("\nPlot saved as 'yelp_sentiment_analysis.png'")

print("\n" + "=" * 60)
print("LIVE SENTIMENT PREDICTIONS")
print("=" * 60)
test_reviews = [
    "Absolutely fantastic food and amazing service!",
    "Terrible experience, cold food and rude waiter.",
    "Food was okay, nothing special about the place.",
    "Best sushi I have ever tasted, will come back!",
    "Waited 2 hours and the food was still undercooked.",
    "Decent pasta but the ambiance was just average.",
]
for review in test_reviews:
    blob  = TextBlob(review)
    pol   = blob.sentiment.polarity
    label = 'Positive' if pol > 0.1 else ('Negative' if pol < -0.1 else 'Neutral')
    icon  = '✅' if label == 'Positive' else ('❌' if label == 'Negative' else '🔶')
    print(f"  {icon} [{label:8s}] ({pol:+.2f})  \"{review}\"")

print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
for s, cnt in df['sentiment'].value_counts().items():
    pct = cnt / len(df) * 100
    print(f"  {s:10s}: {cnt:4d} reviews ({pct:.1f}%)")
print(f"\n  Total Reviews     : {len(df)}")
print(f"  Mean Polarity     : {df['polarity'].mean():.3f}")
print(f"  Mean Subjectivity : {df['subjectivity'].mean():.3f}")
print(f"  Mean Star Rating  : {df['stars'].mean():.2f} / 5.0")
print(f"  Best Restaurant   : {df.groupby('restaurant')['polarity'].mean().idxmax()}")
print(f"  Best Cuisine      : {df.groupby('cuisine')['polarity'].mean().idxmax()}")
print("=" * 60)
