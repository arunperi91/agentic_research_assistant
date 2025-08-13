import asyncio
import aiohttp
from typing import List, Dict, Any, Optional
from duckduckgo_search import AsyncDDGS
import re
from urllib.parse import urlparse, urljoin
import time

class WebSearchService:
    def __init__(self):
        self.ddgs = AsyncDDGS()
        self.session = None
        self.rate_limit_delay = 1.0  # Seconds between requests
        self.last_request_time = 0
    
    async def _ensure_session(self):
        """Ensure aiohttp session exists"""
        if not self.session:
            timeout = aiohttp.ClientTimeout(total=30)
            self.session = aiohttp.ClientSession(timeout=timeout)
    
    async def search(self, query: str, max_results: int = 10) -> List[Dict[str, Any]]:
        """Perform web search with content extraction"""
        try:
            await self._rate_limit()
            
            # Search using DuckDuckGo
            search_results = []
            async for result in self.ddgs.text(query, max_results=max_results):
                search_results.append(result)
            
            # Extract content for each result
            processed_results = []
            for result in search_results:
                try:
                    content = await self.extract_content(result['href'])
                    if content:
                        processed_result = {
                            'title': result.get('title', ''),
                            'url': result.get('href', ''),
                            'content': content,
                            'domain': self._extract_domain(result.get('href', '')),
                            'snippet': result.get('body', '')
                        }
                        processed_results.append(processed_result)
                except Exception as e:
                    print(f"Failed to extract content from {result.get('href', '')}: {e}")
                    continue
            
            return processed_results
            
        except Exception as e:
            print(f"Search failed for query '{query}': {e}")
            return []
    
    async def preview_search(self, query: str, max_results: int = 5) -> List[Dict[str, str]]:
        """Quick search preview without content extraction"""
        try:
            await self._rate_limit()
            
            results = []
            async for result in self.ddgs.text(query, max_results=max_results):
                results.append({
                    'title': result.get('title', ''),
                    'url': result.get('href', ''),
                    'domain': self._extract_domain(result.get('href', '')),
                    'snippet': result.get('body', '')
                })
            
            return results
            
        except Exception as e:
            print(f"Preview search failed for query '{query}': {e}")
            return []
    
    async def extract_content(self, url: str) -> Optional[str]:
        """Extract main content from webpage"""
        try:
            await self._ensure_session()
            await self._rate_limit()
            
            async with self.session.get(url, headers=self._get_headers()) as response:
                if response.status == 200:
                    html = await response.text()
                    content = self._extract_text_from_html(html)
                    return content[:5000]  # Limit content length
                
        except Exception as e:
            print(f"Content extraction failed for {url}: {e}")
            
        return None
    
    def _extract_text_from_html(self, html: str) -> str:
        """Extract readable text from HTML"""
        # Remove script and style elements
        html = re.sub(r'<script[^>]*>.*?</script>', '', html, flags=re.DOTALL | re.IGNORECASE)
        html = re.sub(r'<style[^>]*>.*?</style>', '', html, flags=re.DOTALL | re.IGNORECASE)
        
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', '', html)
        
        # Clean up whitespace
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        
        return text
    
    def _extract_domain(self, url: str) -> str:
        """Extract domain from URL"""
        try:
            parsed = urlparse(url)
            return parsed.netloc
        except Exception:
            return "unknown"
    
    def _get_headers(self) -> Dict[str, str]:
        """Get request headers"""
        return {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
        }
    
    async def _rate_limit(self):
        """Implement rate limiting"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        
        if time_since_last < self.rate_limit_delay:
            await asyncio.sleep(self.rate_limit_delay - time_since_last)
        
        self.last_request_time = time.time()
    
    async def close(self):
        """Close aiohttp session"""
        if self.session:
            await self.session.close()
