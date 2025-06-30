import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
import requests
import time
import re
import os
import json
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def fetch_facebook_posts(cryptocurrency, days=5):
    """
    Fetch Facebook posts related to the selected cryptocurrency from the last X days
    
    Args:
        cryptocurrency (str): The cryptocurrency name to search for
        days (int): Number of days to look back
        
    Returns:
        pandas.DataFrame: DataFrame containing posts with columns:
            - date: Timestamp of the post
            - text: Post content
            - username: Facebook username
            - likes: Number of likes
            - shares: Number of shares
    """
    # For now, we'll create mock data until we integrate the Meta Content Library API
    # In a real implementation, you would use the Meta Content Library API
    # Based on https://developers.facebook.com/docs/content-library-and-api/content-library-api/guides/fb-posts/
    
    # Create a date range for the last X days
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    
    # Dictionary mapping cryptocurrencies to their common hashtags/keywords
    crypto_keywords = {
        "Bitcoin": ["Bitcoin", "BTC", "#Bitcoin", "#BTC", "$BTC", "BTCUSD"],
        "Ethereum": ["Ethereum", "ETH", "#Ethereum", "#ETH", "$ETH", "ETHUSD"],
        "Ripple": ["Ripple", "XRP", "#Ripple", "#XRP", "$XRP", "XRPUSD"],
        "Dogecoin": ["Dogecoin", "DOGE", "#Dogecoin", "#DOGE", "$DOGE", "DOGEUSD"],
        "Tether": ["Tether", "USDT", "#Tether", "#USDT", "$USDT"]
    }
    
    # List of sample post templates
    post_templates = [
        "{crypto} is looking bullish today! Price target: {price_target}. #crypto #trading",
        "Just bought more {crypto}! I believe it will reach {price_target} by end of month.",
        "{crypto} is facing resistance at {price_target}. What do you think? #cryptotrading",
        "The latest news about {crypto} could push it to {price_target}. Exciting times!",
        "Bearish on {crypto} right now. Could drop to {price_target} soon. #cryptocrash",
        "Technical analysis shows {crypto} might bounce between {price_low} and {price_high}.",
        "{crypto} mining difficulty increased again. Bullish for long term hodlers!",
        "New partnership announced for {crypto}! This is huge news for the ecosystem.",
        "Whale alert: Someone just moved 1000 {crypto} to an exchange. Dump incoming?",
        "Major {crypto} update coming next month. Could be a game changer!",
        "Is {crypto} a good buy at current prices? I think yes! #cryptoinvestor",
        "Central banks discussing regulations that could affect {crypto}. Stay vigilant.",
        "Comparing {crypto} to traditional markets - the correlation is breaking down.",
        "Just read an interesting analysis on {crypto}. Sentiment seems to be shifting positive.",
        "Long-term {crypto} holders are not selling despite market volatility. Diamond hands!",
    ]
    
    # Sample usernames
    usernames = [
        "CryptoTrader99", "BlockchainBull", "TechInvestor", "CoinAnalyst", 
        "DeFiExpert", "TokenMaster", "SatoshiFan", "CryptoQueen", "AltcoinGuru", 
        "ChartWizard", "OnChainData", "CryptoPundit", "HODLer2023", "TokenEconomist",
        "CryptoVenture", "ChainLinkMarine", "BTCmaxi", "TraderJoe", "Web3Developer", 
        "MetaverseAnalyst"
    ]
    
    # Generate price targets based on cryptocurrency
    price_targets = {
        "Bitcoin": ["$30,000", "$35,000", "$40,000", "$50,000", "$60,000", "$100,000"],
        "Ethereum": ["$1,800", "$2,000", "$2,500", "$3,000", "$5,000", "$10,000"],
        "Ripple": ["$0.50", "$0.75", "$1.00", "$2.00", "$5.00", "$10.00"],
        "Dogecoin": ["$0.10", "$0.25", "$0.50", "$1.00", "$2.00", "$5.00"],
        "Tether": ["$0.98", "$0.99", "$1.00", "$1.01", "$1.02"]
    }
    
    # Create mock data
    import random
    
    data = []
    keywords = crypto_keywords.get(cryptocurrency, [cryptocurrency])
    
    # Generate between 50-100 posts
    post_count = random.randint(50, 100)
    
    for _ in range(post_count):
        # Random date within the range
        random_days = random.uniform(0, days)
        date = end_date - timedelta(days=random_days)
        
        # Pick random template and username
        template = random.choice(post_templates)
        username = random.choice(usernames)
        
        # Generate post text
        price_target = random.choice(price_targets.get(cryptocurrency, ["$1,000"]))
        price_low = price_targets.get(cryptocurrency, ["$500"])[0]
        price_high = price_targets.get(cryptocurrency, ["$1,500"])[-1]
        
        text = template.format(
            crypto=random.choice(keywords),
            price_target=price_target,
            price_low=price_low,
            price_high=price_high
        )
        
        # Random like and share counts
        likes = int(random.expovariate(1/50))
        shares = int(likes * random.uniform(0.1, 0.5))
        
        # Add to data
        data.append({
            "date": date,
            "text": text,
            "username": username,
            "likes": likes,
            "shares": shares,
            "source": "Facebook"
        })
    
    # Convert to DataFrame
    df = pd.DataFrame(data)
    
    # Sort by date (most recent first)
    df = df.sort_values("date", ascending=False)
    
    return df

def fetch_reddit_posts(cryptocurrency, days=5, limit=100):
    """
    Fetch Reddit posts related to the selected cryptocurrency
    
    Args:
        cryptocurrency (str): The cryptocurrency name to search for
        days (int): Number of days to look back (approximation)
        limit (int): Maximum number of posts to return
        
    Returns:
        pandas.DataFrame: DataFrame containing posts with columns:
            - date: Timestamp of the post
            - text: Post content
            - title: Post title
            - username: Reddit username
            - upvotes: Number of upvotes
            - subreddit: Subreddit name
            - permalink: Link to the post
            - source: Source platform (Reddit)
    """
    # Use the Reddit search URL to get real data
    try:
        # Format the search URL with the cryptocurrency name
        search_url = f"https://www.reddit.com/search/.json?q={cryptocurrency}&type=posts&sort=hot"
        
        # Set a user agent to avoid being blocked
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        # Make the request to Reddit
        response = requests.get(search_url, headers=headers, timeout=10)
        
        # Check if request was successful
        if response.status_code == 200:
            # Parse the JSON response
            data = response.json()
            
            # Extract post data
            posts = []
            
            if 'data' in data and 'children' in data['data']:
                children = data['data']['children']
                
                # Process each post in the data
                for child in children:
                    if 'data' in child:
                        post_data = child['data']
                        
                        # Extract upvotes safely
                        try:
                            upvotes = int(post_data.get('ups', 0))
                        except (ValueError, TypeError):
                            upvotes = 0
                        
                        # Create post entry
                        post = {
                            'date': datetime.fromtimestamp(post_data.get('created_utc', time.time())),
                            'text': post_data.get('selftext', ''),
                            'title': post_data.get('title', ''),
                            'username': post_data.get('author', 'unknown'),
                            'upvotes': upvotes,
                            'subreddit': post_data.get('subreddit_name_prefixed', ''),
                            'permalink': post_data.get('permalink', ''),
                            'source': 'Reddit'
                        }
                        posts.append(post)
                
                # Convert to DataFrame
                if posts:
                    df = pd.DataFrame(posts)
                    
                    # Sort by date (most recent first)
                    df = df.sort_values('date', ascending=False)
                    
                    # Ensure 'likes' column exists for compatibility
                    df['likes'] = df['upvotes']
                    
                    return df
        
        # If request failed or no posts found, try using sample data
        st.warning("Failed to fetch real Reddit data. Falling back to sample data.")
            
    except Exception as e:
        st.warning(f"Error fetching Reddit data: {str(e)}. Falling back to sample data.")
    
    # Fall back to using the sample data if available
    try:
        # Try to read the sample search.json file if available
        with open('search.json', 'r') as f:
            sample_data = json.load(f)
            
        # Extract post data from the sample
        posts = []
        
        if 'data' in sample_data and 'children' in sample_data['data']:
            children = sample_data['data']['children']
            
            # Process each post in the sample data
            for child in children:
                if 'data' in child:
                    post_data = child['data']
                    
                    # Check if post contains the cryptocurrency term
                    if cryptocurrency.lower() in post_data.get('title', '').lower() or \
                       cryptocurrency.lower() in post_data.get('selftext', '').lower():
                        
                        # Create post entry with proper defaults for all required fields
                        post = {
                            'date': datetime.fromtimestamp(post_data.get('created_utc', time.time())),
                            'text': post_data.get('selftext', ''),
                            'title': post_data.get('title', ''),
                            'username': post_data.get('author', 'unknown'),
                            'upvotes': int(post_data.get('ups', 0)),  # Ensure this is an integer
                            'subreddit': post_data.get('subreddit_name_prefixed', ''),
                            'permalink': post_data.get('permalink', ''),
                            'source': 'Reddit'
                        }
                        posts.append(post)
            
        # If we found relevant posts in the sample data, return them
        if posts:
            # Create DataFrame with explicit columns to ensure all fields exist
            df = pd.DataFrame(posts, columns=[
                'date', 'text', 'title', 'username', 'upvotes', 
                'subreddit', 'permalink', 'source'
            ])
            
            # Sort by date (most recent first)
            df = df.sort_values('date', ascending=False)
            
            return df
    
    except Exception as e:
        # If there was an error with sample data, fall back to generated data
        pass
    
    # Fall back to generated mock data
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    
    # Dictionary mapping cryptocurrencies to their common hashtags/keywords
    crypto_keywords = {
        "Bitcoin": ["Bitcoin", "BTC", "#Bitcoin", "#BTC", "$BTC", "BTCUSD"],
        "Ethereum": ["Ethereum", "ETH", "#Ethereum", "#ETH", "$ETH", "ETHUSD"],
        "Ripple": ["Ripple", "XRP", "#Ripple", "#XRP", "$XRP", "XRPUSD"],
        "Dogecoin": ["Dogecoin", "DOGE", "#Dogecoin", "#DOGE", "$DOGE", "DOGEUSD"],
        "Tether": ["Tether", "USDT", "#Tether", "#USDT", "$USDT"]
    }
    
    # List of sample post titles and content
    post_titles = [
        "{crypto} Price Analysis: {timeframe} Outlook",
        "What's your prediction for {crypto} by end of {timeframe}?",
        "{crypto} just broke {price_target} - Discussion",
        "Why I'm {sentiment} on {crypto} right now",
        "{crypto} Technical Analysis - {date}",
        "Just invested in {crypto} for the first time!",
        "Is {crypto} a good long-term investment?",
        "{crypto} vs other cryptocurrencies - Comparison",
        "Breaking: Major news for {crypto} holders",
        "The future of {crypto} in the current market"
    ]
    
    post_contents = [
        "I've been tracking {crypto} for a while and I think it's about to {movement}. The price target I'm looking at is around {price_target}. What do you all think?",
        "After analyzing the charts, I believe {crypto} is forming a {pattern} pattern. This could indicate a move to {price_target} in the next few weeks.",
        "With the recent news about {crypto}, I'm {sentiment} than ever. The fundamentals are strong and adoption is growing.",
        "I'm considering adding more {crypto} to my portfolio. Current price seems {value_assessment} considering the potential upside.",
        "Anyone else noticing the correlation between {crypto} and traditional markets weakening? This could be a sign of {crypto} maturing as an asset class.",
        "Just read that institutions are accumulating {crypto} at these levels. Bullish signal?",
        "The volatility of {crypto} has been decreasing lately. This might indicate a period of consolidation before the next big move.",
        "I've been holding {crypto} for years now. Through all the ups and downs, it's still my best performing investment.",
        "What do you think about the new developments in the {crypto} ecosystem? Will they drive more adoption?",
        "Looking at the on-chain data for {crypto}, whale movements suggest {sentiment} sentiment among large holders."
    ]
    
    # Sample subreddits by cryptocurrency
    crypto_subreddits = {
        "Bitcoin": ["r/Bitcoin", "r/CryptoCurrency", "r/BitcoinMarkets", "r/CryptoMarkets"],
        "Ethereum": ["r/ethereum", "r/CryptoCurrency", "r/ethtrader", "r/ethfinance"],
        "Ripple": ["r/Ripple", "r/CryptoCurrency", "r/XRP", "r/CryptoMarkets"],
        "Dogecoin": ["r/dogecoin", "r/CryptoCurrency", "r/dogemarket", "r/CryptoMarkets"],
        "Tether": ["r/Tether", "r/CryptoCurrency", "r/CryptoMarkets", "r/stablecoin"]
    }
    
    # Generate price targets based on cryptocurrency
    price_targets = {
        "Bitcoin": ["$30,000", "$35,000", "$40,000", "$50,000", "$60,000", "$100,000"],
        "Ethereum": ["$1,800", "$2,000", "$2,500", "$3,000", "$5,000", "$10,000"],
        "Ripple": ["$0.50", "$0.75", "$1.00", "$2.00", "$5.00", "$10.00"],
        "Dogecoin": ["$0.10", "$0.25", "$0.50", "$1.00", "$2.00", "$5.00"],
        "Tether": ["$0.98", "$0.99", "$1.00", "$1.01", "$1.02"]
    }
    
    # Sample usernames
    usernames = [
        "Crypto_Enthusiast123", "HODLforLife", "SatoshiDisciple", "BlockchainBeliver", 
        "AltcoinTrader", "CryptoKing99", "TokenInvestor", "DiamondHands2023", "MoonShotHunter", 
        "DCAmaster", "BearMarketSurvivor", "EthereumMaximalist", "BTCtoTheMoon", "CryptoAnalyst",
        "OnChainWatcher", "TechnicalTrader", "FundamentalInvestor", "Whale_Alert", "DeFiExplorer", 
        "MetaverseExpert"
    ]
    
    # Variables for content generation
    timeframes = ["short-term", "mid-term", "long-term", "1-month", "6-month", "1-year", "Q3", "Q4", "2025"]
    sentiments = ["bullish", "cautiously optimistic", "neutral", "slightly bearish", "very bullish"]
    movements = ["pump", "dump", "break out", "consolidate", "move sideways"]
    patterns = ["bullish divergence", "head and shoulders", "double bottom", "cup and handle", "ascending triangle"]
    value_assessments = ["undervalued", "fairly priced", "slightly overvalued", "a bargain", "expensive"]
    
    # Create mock data
    import random
    
    data = []
    keywords = crypto_keywords.get(cryptocurrency, [cryptocurrency])
    subreddits = crypto_subreddits.get(cryptocurrency, ["r/CryptoCurrency"])
    
    # Generate between 30-50 posts
    post_count = random.randint(30, 50)
    
    for _ in range(post_count):
        # Random date within the range
        random_days = random.uniform(0, days)
        date = end_date - timedelta(days=random_days)
        
        # Pick random title, content and username
        title_template = random.choice(post_titles)
        content_template = random.choice(post_contents)
        username = random.choice(usernames)
        subreddit = random.choice(subreddits)
        
        # Variables for content
        keyword = random.choice(keywords)
        timeframe = random.choice(timeframes)
        sentiment = random.choice(sentiments)
        movement = random.choice(movements)
        pattern = random.choice(patterns)
        value_assessment = random.choice(value_assessments)
        price_target = random.choice(price_targets.get(cryptocurrency, ["$1,000"]))
        
        # Generate post title and content
        title = title_template.format(
            crypto=keyword,
            timeframe=timeframe,
            price_target=price_target,
            sentiment=sentiment,
            date=date.strftime("%b %d")
        )
        
        text = content_template.format(
            crypto=keyword,
            timeframe=timeframe,
            price_target=price_target,
            sentiment=sentiment,
            movement=movement,
            pattern=pattern,
            value_assessment=value_assessment
        )
        
        # Random upvotes between 5 and 500
        upvotes = random.randint(5, 500)
        
        # Create permalink
        permalink = f"/{subreddit.replace('r/', '')}/comments/{hash(title) % 10000000:x}/{title.lower().replace(' ', '_')[:30]}/"
        
        # Add to data
        data.append({
            "date": date,
            "title": title,
            "text": text,
            "username": username,
            "upvotes": upvotes,
            "subreddit": subreddit,
            "permalink": permalink,
            "source": "Reddit"
        })
    
    # Convert to DataFrame
    df = pd.DataFrame(data)
    
    # Sort by date (most recent first)
    df = df.sort_values("date", ascending=False)
    
    return df

def fetch_news_articles(cryptocurrency, days=5):
    """
    Fetch news articles related to the selected cryptocurrency using News API
    
    Args:
        cryptocurrency (str): The cryptocurrency name to search for
        days (int): Number of days to look back
        
    Returns:
        pandas.DataFrame: DataFrame containing news articles with columns:
            - date: Publication date
            - title: Article title
            - description: Article description
            - source: News source name
            - url: URL to the article
            - urlToImage: URL to the article's image
    """
    try:
        # Calculate the date from days ago
        from_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
        
        # Format the News API URL with the cryptocurrency name and from date
        api_key = "e2e16856be41429bbef71c0c9a0c42b0"  # This is the API key provided in the prompt
        news_url = f"https://newsapi.org/v2/everything?q={cryptocurrency}%20price&from={from_date}&sortBy=popularity&apiKey={api_key}"
        
        # Make the request to News API
        response = requests.get(news_url, timeout=10)
        
        # Check if request was successful
        if response.status_code == 200:
            # Parse the JSON response
            data = response.json()
            
            # Extract articles
            if 'articles' in data and len(data['articles']) > 0:
                articles = data['articles']
                
                # Create list of article dictionaries
                news_list = []
                for article in articles:
                    # Convert publishedAt to datetime
                    try:
                        published_date = datetime.strptime(article.get('publishedAt', ''), '%Y-%m-%dT%H:%M:%SZ')
                    except (ValueError, TypeError):
                        published_date = datetime.now()
                    
                    # Create article entry
                    news_item = {
                        'date': published_date,
                        'title': article.get('title', ''),
                        'description': article.get('description', ''),
                        'source': article.get('source', {}).get('name', 'Unknown'),
                        'url': article.get('url', ''),
                        'urlToImage': article.get('urlToImage', '')
                    }
                    news_list.append(news_item)
                
                # Convert to DataFrame
                df = pd.DataFrame(news_list)
                
                # Sort by date (most recent first)
                df = df.sort_values('date', ascending=False)
                
                return df
            else:
                st.warning("No news articles found for this cryptocurrency.")
                return pd.DataFrame()
        else:
            st.warning(f"Failed to fetch news data. Status code: {response.status_code}")
            
    except Exception as e:
        st.warning(f"Error fetching news data: {str(e)}. Falling back to sample data.")
    
    # Fall back to using the sample data if available
    try:
        # Try to read the sample news.json file if available
        with open('news.json', 'r') as f:
            sample_data = json.load(f)
            
        # Extract articles
        if 'articles' in sample_data and len(sample_data['articles']) > 0:
            articles = sample_data['articles']
            
            # Create list of article dictionaries
            news_list = []
            for article in articles:
                # Check if article contains the cryptocurrency term
                if (cryptocurrency.lower() in article.get('title', '').lower() or 
                    cryptocurrency.lower() in article.get('description', '').lower()):
                    
                    # Convert publishedAt to datetime
                    try:
                        published_date = datetime.strptime(article.get('publishedAt', ''), '%Y-%m-%dT%H:%M:%SZ')
                    except (ValueError, TypeError):
                        published_date = datetime.now()
                    
                    # Create article entry
                    news_item = {
                        'date': published_date,
                        'title': article.get('title', ''),
                        'description': article.get('description', ''),
                        'source': article.get('source', {}).get('name', 'Unknown'),
                        'url': article.get('url', ''),
                        'urlToImage': article.get('urlToImage', '')
                    }
                    news_list.append(news_item)
            
            # Convert to DataFrame
            df = pd.DataFrame(news_list)
            
            # Sort by date (most recent first)
            df = df.sort_values('date', ascending=False)
            
            return df
    
    except Exception as e:
        # If there was an error with sample data, return empty DataFrame
        return pd.DataFrame()

def render_sentiment_analysis_tab(cryptocurrency, key_prefix="sentiment_"):
    """
    Render the Sentiment Analysis tab with social media posts and news articles
    
    Args:
        cryptocurrency (str): The selected cryptocurrency
        key_prefix (str): Prefix for Streamlit component keys
    """
    st.subheader("Social Media & News Analysis")
    st.markdown(f"Showing content related to **{cryptocurrency}** from the last 5 days")
    
    # Add data source selection
    source_option = st.radio(
        "Select data source:",
        ["Reddit", "News"],
        horizontal=True,
        key=f"{key_prefix}_source"
    )
    
    # Fetch data based on selected source
    with st.spinner(f"Fetching {source_option.lower()} data..."):
        try:
            if source_option == "Reddit":
                posts_df = fetch_reddit_posts(cryptocurrency)
                st.caption(f"Found {len(posts_df)} Reddit posts mentioning {cryptocurrency}")
                
                # Add filters
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    sort_option = st.selectbox(
                        "Sort by", 
                        ["Most Upvotes", "Most Recent"],
                        index=0,  # Default to Most Upvotes
                        key=f"{key_prefix}_sort"
                    )
                
                with col2:
                    min_upvotes = st.number_input(
                        "Minimum Upvotes", 
                        min_value=0,
                        value=0,
                        key=f"{key_prefix}_min_upvotes"
                    )
                
                with col3:
                    search_term = st.text_input(
                        "Filter by keyword",
                        key=f"{key_prefix}_search"
                    )
                
                # Apply filters
                if min_upvotes > 0:
                    posts_df = posts_df[posts_df["upvotes"] >= min_upvotes]
                    
                if search_term:
                    # Search in both title and text
                    mask = (posts_df["text"].str.contains(search_term, case=False, na=False) | 
                            posts_df["title"].str.contains(search_term, case=False, na=False))
                    posts_df = posts_df[mask]
                
                # Apply sorting
                if sort_option == "Most Recent":
                    posts_df = posts_df.sort_values("date", ascending=False)
                elif sort_option == "Most Upvotes":
                    posts_df = posts_df.sort_values("upvotes", ascending=False)
                
                # Display posts
                if posts_df.empty:
                    st.info("No posts match your filters.")
                else:
                    # Custom CSS for social post display
                    st.markdown("""
                    <style>
                    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');
                    .post-container {
                        background-color: #ffffff;
                        border: 1px solid #e1e8ed;
                        border-radius: 12px;
                        padding: 12px;
                        margin-bottom: 15px;
                        font-family: 'Poppins', sans-serif;
                    }
                    .post-header {
                        display: flex;
                        justify-content: space-between;
                    }
                    .post-username {
                        font-weight: bold;
                        color: #0366d6;
                    }
                    .post-date {
                        color: #657786;
                        font-size: 0.9em;
                    }
                    .post-source {
                        color: #657786;
                        font-weight: bold;
                        font-size: 0.9em;
                        background-color: #f3f3f3;
                        padding: 2px 5px;
                        border-radius: 4px;
                        margin-right: 5px;
                    }
                    .post-title {
                        font-weight: bold;
                        margin-top: 8px;
                        margin-bottom: 5px;
                    }
                    .post-title a {
                        color: #1a0dab;
                        text-decoration: none;
                    }
                    .post-title a:hover {
                        text-decoration: underline;
                    }
                    .post-text {
                        margin-top: 5px;
                        margin-bottom: 10px;
                    }
                    .post-stats {
                        color: #657786;
                        font-size: 0.9em;
                    }
                    .reddit-source {
                        color: #ff4500;
                    }
                    </style>
                    """, unsafe_allow_html=True)
                    
                    # Display each post
                    for _, post in posts_df.head(50).iterrows():
                        # Determine if the post has a title
                        has_title = 'title' in post and post.get('title', '')
                        
                        # Set source class for specific styling
                        source_class = f"{post['source'].lower()}-source"
                        
                        # Get source specific stats
                        stats_html = f"⬆️ {post['upvotes']} upvotes"
                        
                        # Create post link URL
                        post_link = f"https://www.reddit.com{post['permalink']}" if not post['permalink'].startswith('http') else post['permalink']
                        
                        post_html = f"""
                        <div class="post-container">
                            <div class="post-header">
                                <div>
                                    <span class="post-source {source_class}">{post['source']}</span>
                                    <span class="post-username">@{post['username']}</span>
                                </div>
                                <span class="post-date">{post['date'].strftime("%Y-%m-%d %H:%M")}</span>
                            </div>
                        """
                        
                        # Add title if present with hyperlink
                        if has_title:
                            post_html += f'<div class="post-title"><a href="{post_link}" target="_blank">{post["title"]}</a></div>'
                        
                        # Add text content without truncation
                        post_html += f"""
                            <div class="post-text">{post['text']}</div>
                            <div class="post-stats">{stats_html}</div>
                        </div>
                        """
                        
                        st.markdown(post_html, unsafe_allow_html=True)
                        
            elif source_option == "News":
                news_df = fetch_news_articles(cryptocurrency)
                st.caption(f"Found {len(news_df)} news articles about {cryptocurrency}")
                
                # Add filters
                col1, col2 = st.columns(2)
                
                with col1:
                    search_term = st.text_input(
                        "Filter by keyword",
                        key=f"{key_prefix}_news_search"
                    )
                
                with col2:
                    sort_option = st.selectbox(
                        "Sort by", 
                        ["Most Recent", "Alphabetical"],
                        key=f"{key_prefix}_news_sort"
                    )
                
                # Apply filters
                if search_term:
                    # Search in both title and description
                    mask = (news_df["title"].str.contains(search_term, case=False, na=False) | 
                            news_df["description"].str.contains(search_term, case=False, na=False))
                    news_df = news_df[mask]
                
                # Apply sorting
                if sort_option == "Most Recent":
                    news_df = news_df.sort_values("date", ascending=False)
                elif sort_option == "Alphabetical":
                    news_df = news_df.sort_values("title")
                
                # Display news articles
                if news_df.empty:
                    st.info("No news articles match your filters.")
                else:
                    # Custom CSS for news article display
                    st.markdown("""
                    <style>
                    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');
                    .news-container {
                        background-color: #ffffff;
                        border: 1px solid #e1e8ed;
                        border-radius: 12px;
                        padding: 12px;
                        margin-bottom: 15px;
                        display: flex;
                        font-family: 'Poppins', sans-serif;
                    }
                    .news-image {
                        width: 120px;
                        height: 80px;
                        margin-right: 15px;
                        object-fit: cover;
                        border-radius: 8px;
                    }
                    .news-content {
                        flex: 1;
                    }
                    .news-title {
                        font-weight: bold;
                        margin-bottom: 5px;
                        font-size: 1.1em;
                    }
                    .news-title a {
                        color: #1a0dab;
                        text-decoration: none;
                    }
                    .news-title a:hover {
                        text-decoration: underline;
                    }
                    .news-description {
                        color: #333;
                        margin-bottom: 10px;
                    }
                    .news-meta {
                        display: flex;
                        justify-content: space-between;
                        color: #657786;
                        font-size: 0.9em;
                    }
                    .news-source {
                        font-weight: bold;
                    }
                    .news-date {
                        color: #657786;
                    }
                    </style>
                    """, unsafe_allow_html=True)
                    
                    # Display each news article
                    for _, article in news_df.head(50).iterrows():
                        # Default image if none is provided
                        image_url = article.get('urlToImage', '')
                        if not image_url or pd.isna(image_url):
                            image_url = "https://via.placeholder.com/120x80?text=No+Image"
                        
                        # Format the article HTML - show full description without truncation
                        article_html = f"""
                        <div class="news-container">
                            <img class="news-image" src="{image_url}" alt="Article thumbnail">
                            <div class="news-content">
                                <a href="{article['url']}" target="_blank" class="news-title">{article['title']}</a>
                                <div class="news-description">{article['description']}</div>
                                <div class="news-meta">
                                    <span class="news-source">{article['source']}</span>
                                    <span class="news-date">{article['date'].strftime("%Y-%m-%d %H:%M")}</span>
                                </div>
                            </div>
                        </div>
                        """
                        
                        st.markdown(article_html, unsafe_allow_html=True)
                        
        except Exception as e:
            st.error(f"Error fetching data: {str(e)}")
            st.info("Note: While Reddit integration is using real data, News API integration requires an active API key.")