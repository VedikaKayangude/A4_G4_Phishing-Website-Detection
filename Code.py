# -*- coding: utf-8 -*-
"""Phishing URL Detector with Enhanced Features - DATASET OPTIMIZED - FIXED - WITH QR CODE SCANNING"""

!pip install -q gradio scikit-learn tldextract joblib chardet pandas openpyxl requests beautifulsoup4
!pip install -q gradio scikit-learn tldextract joblib chardet pandas openpyxl requests beautifulsoup4 python-whois dnspython
!pip install -q qrcode pillow opencv-python

# ========== INSTALL SYSTEM DEPENDENCIES FOR QR CODE SCANNING ==========
!apt-get update
!apt-get install -y libzbar0

# Now install pyzbar after system dependencies
!pip install -q pyzbar

import os, re, joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from urllib.parse import urlparse, urljoin, quote
import requests
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from bs4 import BeautifulSoup
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
import gradio as gr
import tldextract
import socket
import whois
import ssl
import dns.resolver
from datetime import datetime
import json

# ========== QR CODE SCANNING IMPORTS ==========
import cv2
from pyzbar.pyzbar import decode
from PIL import Image, ImageDraw, ImageFont
import qrcode
import io
import base64

# ========== ALTERNATIVE QR CODE SCANNING FUNCTION ==========
def scan_qr_code_fallback(image):
    """
    Fallback QR code scanning using OpenCV's QRCodeDetector
    """
    try:
        # Convert PIL Image to OpenCV format
        if isinstance(image, Image.Image):
            open_cv_image = np.array(image)
            open_cv_image = open_cv_image[:, :, ::-1].copy()  # Convert RGB to BGR
        else:
            open_cv_image = image

        # Try using OpenCV's built-in QR code detector
        qr_detector = cv2.QRCodeDetector()
        
        data, bbox, _ = qr_detector.detectAndDecode(open_cv_image)
        
        if bbox is not None and data:
            # Draw bounding box
            annotated_image = open_cv_image.copy()
            bbox = bbox.astype(int)
            for i in range(len(bbox)):
                cv2.line(annotated_image, tuple(bbox[i][0]), tuple(bbox[(i+1) % len(bbox)][0]), 
                        color=(0, 255, 0), thickness=3)
            
            # Add text label
            cv2.putText(annotated_image, f"URL: {data[:30]}...", 
                       (bbox[0][0][0], bbox[0][0][1] - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Convert back to PIL
            annotated_image_rgb = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
            annotated_pil = Image.fromarray(annotated_image_rgb)
            
            return True, data, annotated_pil
        else:
            return False, "No QR code found in the image", image
            
    except Exception as e:
        return False, f"Error scanning QR code: {str(e)}", image

def scan_qr_code_advanced(image):
    """
    Advanced QR code scanning that tries multiple methods
    """
    try:
        # First try pyzbar (if available)
        try:
            from pyzbar.pyzbar import decode as pyzbar_decode
            
            if isinstance(image, Image.Image):
                open_cv_image = np.array(image)
                open_cv_image = open_cv_image[:, :, ::-1].copy()
            else:
                open_cv_image = image
                
            decoded_objects = pyzbar_decode(open_cv_image)
            
            if decoded_objects:
                urls_found = []
                annotated_image = open_cv_image.copy()
                
                for obj in decoded_objects:
                    url = obj.data.decode('utf-8')
                    urls_found.append(url)
                    
                    # Draw bounding box
                    points = obj.polygon
                    if len(points) > 4:
                        hull = cv2.convexHull(np.array([point for point in points], dtype=np.float32))
                        hull = hull.astype(int)
                        cv2.polylines(annotated_image, [hull], True, (0, 255, 0), 3)
                    else:
                        points = np.array(points, dtype=np.int32)
                        cv2.polylines(annotated_image, [points], True, (0, 255, 0), 3)
                    
                    cv2.putText(annotated_image, f"URL: {url[:30]}...", 
                               (points[0][0], points[0][1] - 10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                annotated_image_rgb = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
                annotated_pil = Image.fromarray(annotated_image_rgb)
                
                if len(urls_found) == 1:
                    return True, urls_found[0], annotated_pil
                else:
                    return True, f"Multiple URLs found: {', '.join(urls_found)}", annotated_pil
                    
        except ImportError:
            print("pyzbar not available, using OpenCV fallback")
        
        # Fallback to OpenCV QRCodeDetector
        return scan_qr_code_fallback(image)
        
    except Exception as e:
        # Final fallback
        return scan_qr_code_fallback(image)

# ========== DATASET LOADING ==========
import zipfile
import pandas as pd

def load_dataset():
    try:
        print(" Loading dataset from Google Colab...")

        #  Try phi.xlsx in both locations
        for path in ["phi.xlsx", "/content/phi.xlsx"]:
            if os.path.exists(path):
                print(f" Found your dataset: {path}")
                try:
                    df = pd.read_excel(path, engine="openpyxl")
                    print(f" Loaded {len(df)} samples from {path}")
                    print(f" Dataset columns: {df.columns.tolist()}")
                    print(f" Unique labels: {df[df.columns[1]].unique()}")
                    return df
                except Exception as e:
                    print(f" Could not read {path}: {e}")

        #  Fallback: search
        if os.path.isdir("/content/"):
            for f in os.listdir("/content/"):
                if f.lower().endswith((".xlsx", ".csv")):
                    print(f" Trying to load {f}")
                    full = os.path.join("/content", f)
                    try:
                        if f.endswith(".xlsx"):
                            df = pd.read_excel(full, engine="openpyxl")
                        else:
                            df = pd.read_csv(full)
                        print(f" Loaded {len(df)} samples from {f}")
                        print(f" Dataset columns: {df.columns.tolist()}")
                        print(f" Unique labels: {df[df.columns[1]].unique()}")
                        return df
                    except Exception as e:
                        print(f" Skipped {f}: {e}")

        #  Dummy fallback
        print(" Creating enhanced dummy dataset...")
        return create_enhanced_dummy_data()

    except Exception as e:
        print(f" Error loading dataset: {e}")
        return create_enhanced_dummy_data()

def create_enhanced_dummy_data():
    """Create dummy dataset for testing"""
    data = {
        'url': [
            'https://google.com',
            'https://github.com',
            'https://facebook.com',
            'http://secure-bank-verify.com',
            'http://goooogle.com/login',
            'http://facebo0k.com',
            'https://paypal.com',
            'http://paypa1-security.com',
            'https://amazon.com',
            'http://amaz0n-account.com',
            'http://www.designeremdoces.com/components/com_contact/ggdrives/',
            'http://secure-login-bank-account.com',
            'http://update-payment-info.com',
            'https://en.wikipedia.org/wiki/North_Dakota',
            'https://example.com'
        ],
        'label': [
            'benign', 'benign', 'benign', 'phishing', 'phishing',
            'phishing', 'benign', 'phishing', 'benign', 'phishing',
            'phishing', 'phishing', 'phishing', 'benign', 'benign'
        ]
    }
    return pd.DataFrame(data)

# ========== TRUSTED DOMAINS LIST ==========
TRUSTED_DOMAINS = {
    # Social Media
    "facebook.com", "whatsapp.com", "twitter.com", "instagram.com", "linkedin.com",
    "youtube.com", "tiktok.com", "pinterest.com", "reddit.com", "snapchat.com",

    # Tech & Cloud
    "google.com", "microsoft.com", "apple.com", "amazon.com", "netflix.com",
    "github.com", "stackoverflow.com", "mozilla.org", "wikipedia.org",

    # Banking & Finance
    "paypal.com", "stripe.com", "visa.com", "mastercard.com", "americanexpress.com",

    # Common Services
    "wordpress.com", "blogspot.com", "medium.com", "quora.com", "spotify.com",

    # Government & Education
    "gov", "edu", "org", "mil"
}

# ========== VALID TLDs ==========
VALID_TLDS = {
    'com', 'org', 'net', 'edu', 'gov', 'mil', 'io', 'co', 'info', 'biz',
    'me', 'tv', 'us', 'uk', 'ca', 'au', 'de', 'fr', 'jp', 'cn', 'in',
    'it', 'es', 'nl', 'br', 'ru', 'za', 'mx', 'ar', 'ch', 'se', 'no', 'dk', 'fi'
}

# ========== SUSPICIOUS URL PATTERNS ==========
SUSPICIOUS_KEYWORDS = {
    'login', 'verify', 'secure', 'account', 'banking', 'update', 'confirm', 'validation',
    'authenticate', 'password', 'credential', 'signin', 'signup', 'wallet', 'payment',
    'billing', 'invoice', 'security', 'admin', 'support', 'service', 'alert', 'urgent',
    'important', 'action', 'required', 'suspend', 'limited', 'verify', 'recover',
    'unlock', 'restore', 'claim', 'reward', 'bonus', 'free', 'winner', 'prize',
    'lottery', 'selected', 'congratulations', 'offer', 'discount', 'limited-time'
}

# ========== CONTENT ANALYSIS - MORE SPECIFIC PATTERNS ==========
CONTENT_SUSPICIOUS_KEYWORDS = {
    'verify your account', 'update your payment', 'confirm your identity',
    'password required', 'credit card number', 'social security number',
    'account suspended', 'unusual activity', 'bank account', 'password reset',
    'security verification', 'login verification', 'account verification',
    'billing information', 'payment method', 'card number', 'expiration date',
    'cvv', 'routing number', 'account number'
}

# ========== CRAWLER CONFIGURATION ==========
REQUEST_HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
    'Accept-Language': 'en-US,en;q=0.9',
}

# ========== ENHANCED CONTENT CRAWLER FUNCTION ==========
def extract_website_content(url):
    """
    Fetches the URL content and extracts all text content.
    Returns a tuple: (website_text, content_flags, suspicious_text_found)
    """
    content_flags = []
    website_text = ""
    suspicious_text_found = []

    try:
        # Use a timeout to avoid hanging
        response = requests.get(url, headers=REQUEST_HEADERS, timeout=10, allow_redirects=True)
        response.raise_for_status()

        # Check if content is HTML
        content_type = response.headers.get('Content-Type', '').lower()
        if 'text/html' not in content_type:
            website_text = f"Non-HTML content detected. Content-Type: {content_type}"
            return website_text, content_flags, suspicious_text_found

        soup = BeautifulSoup(response.text, 'html.parser')

        # Remove script and style elements
        for script in soup(["script", "style", "meta", "link"]):
            script.decompose()

        # Extract all text content with better formatting
        website_text = soup.get_text()
        # Clean up the text
        lines = (line.strip() for line in website_text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        website_text = '\n'.join(chunk for chunk in chunks if chunk)

        page_text_lower = website_text.lower()

        # Get base domain for comparison
        ext = tldextract.extract(url)
        base_domain = f"{ext.domain}.{ext.suffix}"

        # 1. Check for suspicious forms
        forms = soup.find_all('form')
        if forms:
            for form in forms:
                action = form.get('action', '')
                absolute_action_url = urljoin(url, action)
                action_ext = tldextract.extract(absolute_action_url)
                action_domain = f"{action_ext.domain}.{action_ext.suffix}"

                if action_domain != base_domain and action_domain:
                    content_flags.append(f"Form submits to external domain: {action_domain}")

        # 2. Check for suspicious text with better matching
        for keyword in CONTENT_SUSPICIOUS_KEYWORDS:
            if keyword in page_text_lower:
                content_flags.append(f"Suspicious text: '{keyword}'")
                # Find the actual text with context
                start_idx = max(0, page_text_lower.find(keyword) - 50)
                end_idx = min(len(page_text_lower), page_text_lower.find(keyword) + len(keyword) + 50)
                context = website_text[start_idx:end_idx].strip()
                suspicious_text_found.append(f"'{keyword}'\nContext: ...{context}...")

        # 3. Check for iframes
        iframes = soup.find_all('iframe')
        if iframes:
            content_flags.append(f"Page uses {len(iframes)} iframes")

        # 4. Check for password fields
        password_fields = soup.find_all('input', {'type': 'password'})
        if password_fields:
            content_flags.append(f"Password input fields detected: {len(password_fields)}")

        return website_text, content_flags, suspicious_text_found

    except requests.exceptions.Timeout:
        website_text = "Connection Timeout - Could not fetch website content within 10 seconds"
        content_flags.append("Connection Timeout")
        return website_text, content_flags, suspicious_text_found

    except requests.exceptions.RequestException as e:
        error_msg = f" Connection Error - {str(e)}"
        website_text = error_msg
        content_flags.append(f"Connection Error: {str(e)}")
        return website_text, content_flags, suspicious_text_found

    except Exception as e:
        error_msg = f" Content parsing error: {str(e)}"
        website_text = error_msg
        content_flags.append(f"Parsing error: {str(e)}")
        return website_text, content_flags, suspicious_text_found

# ========== ENHANCED FEATURE EXTRACTION ==========
def extract_url_features(url):
    """Enhanced feature extraction optimized for dataset performance - FIXED"""
    try:
        if not isinstance(url, str) or not url.strip():
            return [0] * 16

        url_strip = url.strip().lower()

        # Basic URL parsing with fallback
        try:
            parsed = urlparse(url_strip)
        except:
            parsed = type('obj', (object,), {'scheme': '', 'netloc': '', 'path': ''})

        # Check for protocol typos - LESS AGGRESSIVE
        protocol_typos = 0
        protocol_errors = ['htttp', 'htttps', 'htps', 'httpss', 'http//', 'https//', 'htp:', 'http:']
        if any(url_strip.startswith(error) for error in protocol_errors):
            protocol_typos = 1

        # Extract domain using tldextract with error handling
        try:
            ext = tldextract.extract(url_strip)
            domain = f"{ext.domain}.{ext.suffix}" if ext.suffix else ext.domain

            # TLD validation - LESS STRICT
            has_valid_tld = ext.suffix in VALID_TLDS if ext.suffix else True
            invalid_tld = 0 if has_valid_tld else 1

            # Suspicious TLD patterns - LESS AGGRESSIVE
            suspicious_tld_patterns = 0
            if ext.suffix:
                if len(ext.suffix) > 6:
                    suspicious_tld_patterns = 1
                if re.search(r'\d', ext.suffix):
                    suspicious_tld_patterns = 1
                if re.search(r'[^a-z.]', ext.suffix):
                    suspicious_tld_patterns = 1

            # Check if domain is trusted
            is_trusted = 0
            for trusted_domain in TRUSTED_DOMAINS:
                if trusted_domain in domain or trusted_domain in url_strip:
                    is_trusted = 1
                    break

        except:
            invalid_tld = 0
            suspicious_tld_patterns = 0
            is_trusted = 0

        # Basic features
        url_length = len(url_strip)
        num_dots = url_strip.count('.')
        num_hyphens = url_strip.count('-')
        num_digits = sum(c.isdigit() for c in url_strip)
        num_params = url_strip.count('?') + url_strip.count('&') + url_strip.count('=')

        # Security features with safe parsing
        has_ip = 0
        try:
            netloc_part = parsed.netloc.split(':')[0] if hasattr(parsed, 'netloc') and parsed.netloc else ''
            if re.match(r'^\d{1,3}(\.\d{1,3}){3}(:\d+)?$', netloc_part):
                has_ip = 1
            elif re.match(r'^(\d{1,3}\.){3,}', url_strip):
                has_ip = 1
        except:
            has_ip = 0

        has_https = 1 if hasattr(parsed, 'scheme') and parsed.scheme == "https" else 0
        has_at_symbol = 1 if '@' in url_strip else 0

        # Enhanced suspicious patterns - LESS SENSITIVE
        suspicious_keywords_count = 0
        for keyword in SUSPICIOUS_KEYWORDS:
            if keyword in url_strip:
                suspicious_keywords_count += 1

        # Additional phishing indicators
        consecutive_chars = 1 if re.search(r'(.)\1{4,}', url_strip) else 0
        hex_chars = 1 if re.search(r'%[0-9a-f]{2}', url_strip) else 0

        # FIXED: Path-based suspicious patterns - IGNORE TRAILING SLASH
        suspicious_path = 0
        suspicious_path_count = 0
        if hasattr(parsed, 'path'):
            # Normalize path - remove trailing slash for analysis
            path_for_analysis = parsed.path.rstrip('/')
            
            suspicious_path_indicators = ['login', 'admin', 'secure', 'verify', 'account', 'password', 'bank',
                                         'components', 'com_contact', 'ggdrives', 'wp-content', 'wp-admin',
                                         'includes', 'modules', 'templates']
            
            if path_for_analysis:  # Only check if there's an actual path (not just '/')
                path_lower = path_for_analysis.lower()
                for indicator in suspicious_path_indicators:
                    if indicator in path_lower:
                        suspicious_path_count += 1
                suspicious_path = 1 if suspicious_path_count > 2 else 0

        # FIXED: Subdomain analysis - DON'T COUNT 'www' AS SUSPICIOUS
        subdomain_count = 0
        try:
            if hasattr(parsed, 'netloc') and parsed.netloc:
                # Get all parts of netloc
                netloc_parts = parsed.netloc.split('.')
                
                # Count subdomains excluding 'www'
                if len(netloc_parts) > 2:
                    # If we have www.domain.tld, count as 0 subdomains for common sites
                    if netloc_parts[0] == 'www' and len(netloc_parts) == 3:
                        subdomain_count = 0
                    else:
                        subdomain_count = len(netloc_parts) - 2
                else:
                    subdomain_count = 0
                    
                if subdomain_count < 0:
                    subdomain_count = 0
        except:
            subdomain_count = 0

        return [
            url_length, num_dots, num_hyphens, num_digits, num_params,
            has_ip, has_https, has_at_symbol, suspicious_keywords_count, is_trusted,
            protocol_typos, invalid_tld, suspicious_tld_patterns, consecutive_chars,
            suspicious_path, subdomain_count
        ]

    except Exception as e:
        print(f"Error extracting features from {url}: {e}")
        return [0] * 16

ENHANCED_FEATURE_NAMES = [
    "url_length", "num_dots", "num_hyphens", "num_digits", "num_params",
    "has_ip", "has_https", "has_at_symbol", "suspicious_keywords", "is_trusted",
    "protocol_typos", "invalid_tld", "suspicious_tld_patterns", "consecutive_chars",
    "suspicious_path", "subdomain_count"
]

# ========== SECURITY & NETWORK ANALYSIS FUNCTIONS ==========
def get_ip_geolocation(ip_address):
    """Get IP geolocation information"""
    try:
        response = requests.get(f"http://ip-api.com/json/{ip_address}", timeout=5)
        if response.status_code == 200:
            data = response.json()
            return {
                'country': data.get('country', 'Unknown'),
                'isp': data.get('isp', 'Unknown'),
                'org': data.get('org', 'Unknown'),
                'asn': data.get('as', 'Unknown')
            }
    except:
        pass
    return {'country': 'Unknown', 'isp': 'Unknown', 'org': 'Unknown', 'asn': 'Unknown'}

def get_dns_records(domain):
    """Get DNS records for a domain"""
    records = {}
    try:
        # A records
        a_records = dns.resolver.resolve(domain, 'A')
        records['A'] = [str(ip) for ip in a_records]
    except:
        records['A'] = []

    try:
        # MX records
        mx_records = dns.resolver.resolve(domain, 'MX')
        records['MX'] = [str(mx) for mx in mx_records]
    except:
        records['MX'] = []

    return records

def analyze_network_information(url):
    """
    Comprehensive network analysis for the given URL
    """
    try:
        parsed = urlparse(url)
        domain = parsed.netloc

        # Remove port if present
        if ':' in domain:
            domain = domain.split(':')[0]

        network_info = {
            'domain': domain,
            'dns_records': {},
            'ip_analysis': []
        }

        # Get DNS A records
        dns_info = get_dns_records(domain)
        network_info['dns_records'] = dns_info

        # Analyze IP addresses
        if dns_info.get('A'):
            for ip in dns_info['A'][:2]:  # Limit to first 2 IPs
                geo_info = get_ip_geolocation(ip)
                network_info['ip_analysis'].append({
                    'ip': ip,
                    'geolocation': geo_info
                })

        return network_info

    except Exception as e:
        return {'error': f"Network analysis failed: {str(e)}"}

def format_network_report(network_info):
    """Format network analysis into a readable report"""
    if 'error' in network_info:
        return f" Network Analysis Error: {network_info['error']}"

    report = []
    report.append("Network Security Analysis")
    report.append("")

    # DNS Information
    dns_records = network_info.get('dns_records', {})
    if dns_records.get('A'):
        report.append("DNS Records:")
        report.append(f"  • A Records: {', '.join(dns_records['A'])}")

    # IP Analysis
    ip_analysis = network_info.get('ip_analysis', [])
    if ip_analysis:
        report.append("")
        report.append("IP Address Analysis:")
        for ip_data in ip_analysis:
            geo = ip_data.get('geolocation', {})
            report.append(f"  IP: {ip_data.get('ip', 'Unknown')}")
            report.append(f"  Country: {geo.get('country', 'Unknown')}")
            report.append(f"  ISP: {geo.get('isp', 'Unknown')}")

    return "\n".join(report)

# ========== COMPREHENSIVE URL VALIDATION ==========
def validate_url_structure(url):
    """Comprehensive URL validation for phishing detection - FIXED"""
    try:
        url_lower = url.lower().strip()
        red_flags = []

        # Basic URL sanity check
        if not url_lower or len(url_lower) < 4:
            return ["URL is too short or empty"], False

        # 1. Protocol anomalies
        if url_lower.startswith(('htttp', 'htttps', 'htps', 'httpss', 'http//', 'https//')):
            red_flags.append("Suspicious protocol spelling")

        # 2. Extract domain properly with error handling
        try:
            ext = tldextract.extract(url_lower)
            domain = f"{ext.domain}.{ext.suffix}" if ext.suffix else ext.domain

            # 3. TLD validation - LESS STRICT
            if not ext.suffix:
                red_flags.append("Missing TLD")
            elif ext.suffix not in VALID_TLDS:
                red_flags.append(f"Uncommon TLD: {ext.suffix}")  # Changed from "Invalid" to "Uncommon"

            # 4. Suspicious TLD patterns - LESS AGGRESSIVE
            if ext.suffix and len(ext.suffix) > 8:  # Increased threshold
                red_flags.append("Unusually long TLD")
            if ext.suffix and re.search(r'\d', ext.suffix):
                red_flags.append("TLD contains numbers")

            # 5. Domain length anomalies
            if len(domain) > 100:  # Increased threshold
                red_flags.append("Excessively long domain")

            # 6. Check for trusted domains
            is_trusted = any(trusted in domain or trusted in url_lower for trusted in TRUSTED_DOMAINS)

            # 7. Check for obvious phishing patterns
            if re.search(r'([a-zA-Z])\1{4,}', domain):  # Increased to 4+ consecutive chars
                red_flags.append("Suspicious character repetition")

            if '-' in domain and domain.count('-') > 4:  # Increased threshold
                red_flags.append("Too many hyphens in domain")

            if re.search(r'\d{8,}', url_lower):  # Increased threshold
                red_flags.append("Suspicious number sequence")

        except Exception as e:
            red_flags.append(f"Domain extraction error: {str(e)}")
            is_trusted = False

        # 8. Check for URL encoding attempts
        if url_lower.count('%') > 5:  # Only flag excessive encoding
            red_flags.append("Excessive URL encoding detected")

        return red_flags, is_trusted

    except Exception as e:
        return [f"URL validation error: {str(e)}"], False

# ========== DATA PREPARATION ==========
def prepare_features_from_df(df_raw):
    """Optimized feature preparation for dataset performance"""
    # Quick column detection
    url_col = df_raw.columns[0]
    label_col = df_raw.columns[1] if len(df_raw.columns) > 1 else None

    for c in df_raw.columns:
        c_lower = c.lower()
        if 'url' in c_lower or 'link' in c_lower:
            url_col = c
        if 'label' in c_lower or 'type' in c_lower or 'class' in c_lower or 'category' in c_lower:
            label_col = c

    if label_col is None:
        df_raw = df_raw.copy()
        df_raw['label'] = 'benign'
        label_col = 'label'

    # Take samples for training
    df_sample = df_raw.head(50000).copy()
    df_sample = df_sample[[url_col, label_col]]
    df_sample.columns = ['url', 'label']

    # IMPROVED LABEL STANDARDIZATION
    print(" Original labels in dataset:", df_sample['label'].unique())

    def standardize_label(label):
        label_str = str(label).lower().strip()

        # Handle phishing labels
        if any(phish in label_str for phish in ['phishing', 'phish', 'malicious', 'bad', '1', 'unsafe','malware']):
            return 'phishing'
        # Handle defacement labels - treat as HIGH RISK
        elif any(deface in label_str for deface in ['defacement', 'deface', 'vandal', 'hack']):
            return 'phishing'  # Treat defacement as phishing since both are malicious
        # Handle benign labels
        elif any(benign in label_str for benign in ['benign', 'good', 'legitimate', '0', 'safe', 'normal']):
            return 'benign'
        else:
            print(f"⚠️ Unknown label: {label_str}, defaulting to benign")
            return 'benign'

    df_sample['label'] = df_sample['label'].apply(standardize_label)
    print("Standardized labels distribution:")
    print(df_sample['label'].value_counts())

    # Enhanced feature extraction
    print(" Extracting features from URLs...")
    features_list = []
    for i, url in enumerate(df_sample['url']):
        if i % 1000 == 0 and i > 0:
            print(f"  Processed {i}/{len(df_sample)} URLs")
        features_list.append(extract_url_features(url))

    feature_df = pd.DataFrame(features_list, columns=ENHANCED_FEATURE_NAMES)
    result_df = pd.concat([df_sample[['url', 'label']], feature_df], axis=1)

    return result_df

def prepare_Xy(df):
    """Improved data preparation with better label handling"""
    # Print label distribution for debugging
    print("Label distribution in prepared data:")
    label_counts = df['label'].value_counts()
    print(label_counts)

    df_filtered = df[df['label'].isin(['benign', 'phishing'])].copy()

    if len(df_filtered) < 10:
        print("⚠️ Not enough samples with proper labels, using dummy data")
        df_filtered = create_enhanced_dummy_data()
        df_filtered = prepare_features_from_df(df_filtered)

    X = df_filtered[ENHANCED_FEATURE_NAMES].fillna(0)
    y = df_filtered['label'].map({'benign': 0, 'phishing': 1})

    print(f" Final training data: {X.shape[0]} samples")
    print(f"Phishing samples: {y.sum()}")
    print(f"Benign samples: {len(y) - y.sum()}")

    # Check if we have enough phishing samples
    if y.sum() < 5:
        print(" WARNING: Very few phishing samples! Model may not learn properly.")

    return X, y, ENHANCED_FEATURE_NAMES, {'benign': 0, 'phishing': 1}

# ========== MODEL BUILDING ==========
def build_model(model_name):
    models = {
        "KNN": KNeighborsClassifier(n_neighbors=3, weights='distance'),
        "IsolationForest": IsolationForest(
            contamination=0.3,
            random_state=42,
            n_estimators=150,
            max_samples='auto'
        ),
        "DecisionTree": DecisionTreeClassifier(
            random_state=42,
            max_depth=15,
            min_samples_split=2,
            min_samples_leaf=1,
            class_weight='balanced'
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=100,
            max_depth=15,
            random_state=42,
            class_weight='balanced'
        )
    }
    return models.get(model_name, models["RandomForest"])

# ========== LOAD AND PREPARE DATA ==========
print(" Loading dataset from Google Colab...")
df_raw = load_dataset()
print(f" Original dataset shape: {df_raw.shape}")

print("🔧 Preparing features...")
df_features = prepare_features_from_df(df_raw)
print(f" Prepared {len(df_features)} samples with {len(ENHANCED_FEATURE_NAMES)} enhanced features")

def prepare_Xy(df):
    """Fast data preparation"""
    df_filtered = df[df['label'].isin(['benign', 'phishing'])].copy()

    if len(df_filtered) < 2:
        df_filtered = create_enhanced_dummy_data()
        df_filtered = prepare_features_from_df(df_filtered)

    X = df_filtered[ENHANCED_FEATURE_NAMES].fillna(0)
    y = df_filtered['label'].map({'benign': 0, 'phishing': 1})

    return X, y, ENHANCED_FEATURE_NAMES, {'benign': 0, 'phishing': 1}

# Initialize pipeline
X_all, y_all, feature_cols, label_map = prepare_Xy(df_features)
GLOBAL_PIPELINE = {"model": None, "scaler": None, "features": feature_cols, "label_map": label_map}
print(" System ready! Enhanced features:", feature_cols)

# ========== TRAINING FUNCTION ==========
def train_pipeline(model_name):
    """Optimized training function for dataset performance"""
    try:
        start_time = pd.Timestamp.now()

        if X_all.shape[0] < 10:
            return "Need at least 10 samples to train", None, None, "Error"

        print(f" Dataset Info:")
        print(f"  Total samples: {X_all.shape[0]}")
        print(f"   Features: {X_all.shape[1]}")
        print(f"   Phishing ratio: {y_all.mean():.2%}")

        # Use 80-20 split for better accuracy
        X_train, X_test, y_train, y_test = train_test_split(
            X_all, y_all, test_size=0.2, random_state=42, stratify=y_all
        )

        print(f" Training set: {X_train.shape[0]} samples ({y_train.mean():.2%} phishing)")
        print(f" Test set: {X_test.shape[0]} samples ({y_test.mean():.2%} phishing)")

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        model = build_model(model_name)

        # Handle unsupervised vs supervised training
        if isinstance(model, IsolationForest):
            print(f"🔍 Training unsupervised model: {model_name}")
            model.fit(X_train_scaled)

            y_pred = model.predict(X_test_scaled)
            y_pred_binary = [1 if x == -1 else 0 for x in y_pred]
            acc = accuracy_score(y_test, y_pred_binary)

        else:
            print(f" Training supervised model: {model_name}")
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            acc = accuracy_score(y_test, y_pred)
            y_pred_binary = y_pred

        # Detailed performance analysis
        print("\n Detailed Classification Report:")
        print(classification_report(y_test, y_pred_binary, target_names=['Benign', 'Phishing']))

        cm = confusion_matrix(y_test, y_pred_binary)
        print(f"Confusion Matrix:\n{cm}")

        # Calculate precision and recall for phishing class
        if cm.shape == (2, 2):
            phishing_precision = cm[1,1] / (cm[1,1] + cm[0,1]) if (cm[1,1] + cm[0,1]) > 0 else 0
            phishing_recall = cm[1,1] / (cm[1,1] + cm[1,0]) if (cm[1,1] + cm[1,0]) > 0 else 0
            print(f"Phishing Precision: {phishing_precision:.2%}")
            print(f"Phishing Recall: {phishing_recall:.2%}")

        # Confusion Matrix
        fig_cm, ax = plt.subplots(figsize=(5, 4))
        im = ax.imshow(cm, interpolation='nearest', cmap='RdYlGn_r')
        ax.figure.colorbar(im, ax=ax)
        ax.set(xticks=[0, 1], yticks=[0, 1],
               xticklabels=['Benign', 'Phishing'], yticklabels=['Benign', 'Phishing'],
               title=f'Confusion Matrix - {model_name}\nAccuracy: {acc:.2%}',
               ylabel='True Label', xlabel='Predicted Label')

        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], 'd'),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black",
                        fontsize=14, fontweight='bold')
        plt.tight_layout()

        # Training Statistics
        fig_stats, ax2 = plt.subplots(figsize=(6, 4))

        stats_data = [
            X_train.shape[0],
            X_test.shape[0],
            acc * 100
        ]
        stats_labels = ['Training Samples', 'Test Samples', 'Accuracy (%)']
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']

        bars = ax2.bar(stats_labels, stats_data, color=colors, alpha=0.8)
        ax2.set_title(f'Training Statistics - {model_name}', fontweight='bold')
        ax2.set_ylabel('Count / Percentage')

        for bar, value in zip(bars, stats_data):
            height = bar.get_height()
            if stats_labels[bars.index(bar)] == 'Accuracy (%)':
                ax2.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                         f'{value:.2f}%', ha='center', va='bottom', fontweight='bold')
            else:
                ax2.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                         f'{value:.0f}', ha='center', va='bottom', fontweight='bold')
        plt.tight_layout()

        # Save pipeline
        GLOBAL_PIPELINE.update({"model": model, "scaler": scaler})
        joblib.dump(GLOBAL_PIPELINE, "trained_pipeline.joblib")

        training_time = (pd.Timestamp.now() - start_time).total_seconds()

        # Test the model on dataset samples
        test_results = test_model_on_dataset_samples()

        # Simple table format
        summary = f"""
| Metric | Value |
|--------|-------|
| Model | {model_name} |
| Accuracy | {acc:.2%} |
| Training Samples | {X_train.shape[0]} |
| Test Samples | {X_test.shape[0]} |
| Training Time | {training_time:.2f}s |
| Status | READY |
"""

        info = f"Training completed in {training_time:.2f} seconds! Model accuracy: {acc:.2%}"

        return summary, fig_cm, fig_stats, info

    except Exception as e:
        error_msg = f"Training error: {str(e)}"
        import traceback
        print(f"Detailed error: {traceback.format_exc()}")
        return error_msg, None, None, "Error"

# ========== TEST FUNCTION ==========
def test_model_on_dataset_samples():
    """Comprehensive testing on dataset samples"""
    if GLOBAL_PIPELINE['model'] is None:
        print(" No model trained yet!")
        return

    # Test with more samples from each category
    test_urls = []
    test_labels = []

    # Get samples from your dataset
    if 'phishing' in df_features['label'].values:
        phishing_samples = df_features[df_features['label'] == 'phishing'].head(15)['url'].tolist()
        test_urls.extend(phishing_samples)
        test_labels.extend(['PHISHING'] * len(phishing_samples))
        print(f"Testing with {len(phishing_samples)} phishing samples")

    if 'benign' in df_features['label'].values:
        benign_samples = df_features[df_features['label'] == 'benign'].head(15)['url'].tolist()
        test_urls.extend(benign_samples)
        test_labels.extend(['BENIGN'] * len(benign_samples))
        print(f" Testing with {len(benign_samples)} benign samples")

    print("\nCOMPREHENSIVE MODEL TESTING ON DATASET SAMPLES:")
    print("=" * 60)

    correct = 0
    total = len(test_urls)
    results = []

    for i, (test_url, expected_label) in enumerate(zip(test_urls, test_labels)):
        try:
            # Extract features
            features = extract_url_features(test_url)
            X = pd.DataFrame([features], columns=ENHANCED_FEATURE_NAMES)
            X_scaled = GLOBAL_PIPELINE['scaler'].transform(X) if GLOBAL_PIPELINE['scaler'] else X

            # Predict
            if isinstance(GLOBAL_PIPELINE['model'], IsolationForest):
                prediction = GLOBAL_PIPELINE['model'].predict(X_scaled)[0]
                predicted_label = "PHISHING" if prediction == -1 else "BENIGN"
            else:
                prediction = GLOBAL_PIPELINE['model'].predict(X_scaled)[0]
                predicted_label = "PHISHING" if prediction == 1 else "BENIGN"

            status = "" if predicted_label == expected_label else ""
            correct += 1 if predicted_label == expected_label else 0

            result = {
                'url': test_url,
                'expected': expected_label,
                'predicted': predicted_label,
                'correct': predicted_label == expected_label
            }
            results.append(result)

            print(f"{status} Sample {i+1}: {test_url[:60]}...")
            print(f"   Expected: {expected_label}, Got: {predicted_label}")

        except Exception as e:
            print(f" Error testing sample {i+1}: {e}")

    accuracy = correct / total if total > 0 else 0
    print("=" * 60)
    print(f"FINAL TEST RESULTS:")
    print(f"   Accuracy: {accuracy:.1%} ({correct}/{total} correct)")

    if accuracy < 0.8:
        print(" WARNING: Model performance on dataset samples is LOW!")
        print("   The model is not correctly identifying URLs from your dataset.")
    else:
        print(" EXCELLENT: Model is performing well on dataset samples!")

    return results

# ========== SCREENSHOT FUNCTION ==========
def get_screenshot_url(url):
    """Generates a URL to safely screenshot a website using thum.io"""
    try:
        image_url = f"https://image.thum.io/get/width/800/crop/1000/q/90/{url}"
        return image_url
    except Exception as e:
        print(f"Error creating screenshot URL: {e}")
        return None

# ========== COMBINED SCORING FUNCTION - FIXED ==========
def calculate_combined_score(url_features, content_flags, suspicious_text_found, is_trusted_domain, lexical_flags):
    """
    Calculate combined phishing score from URL + Content analysis - FIXED THRESHOLDS
    """
    url_score = 0.0
    content_score = 0.0

    # 1. URL-based scoring (0-70 points) - LESS AGGRESSIVE
    url_risk_factors = [
        url_features[5],  # has_ip
        url_features[7],  # has_at_symbol
        url_features[8] > 3,  # suspicious_keywords > 3 (increased threshold)
        url_features[10],  # protocol_typos
        url_features[11],  # invalid_tld
        url_features[12],  # suspicious_tld_patterns
        url_features[13],  # consecutive_chars
        url_features[14],  # suspicious_path
        len(lexical_flags) > 2,  # More than 2 lexical flags
        url_features[0] > 100,  # Very long URL (increased threshold)
        url_features[3] > 8,  # Too many digits (increased threshold)
    ]

    # Calculate URL score with weights - LOWER WEIGHTS
    url_weights = [1.0, 0.8, 1.5, 1.0, 2.0, 1.5, 0.8, 2.0, 1.5, 0.3, 0.3]  # Reduced weights
    weighted_url_score = sum(weight * factor for weight, factor in zip(url_weights, url_risk_factors))
    url_score = min(0.7, weighted_url_score / sum(url_weights) * 0.7)

    # 2. Content-based scoring (0-30 points)
    if content_flags:
        content_risk_factors = [
            len(content_flags) > 2,  # Increased threshold
            len(suspicious_text_found) > 1,  # Increased threshold
            any('password' in flag.lower() for flag in content_flags),
            any('external domain' in flag.lower() for flag in content_flags),
        ]
        content_score = sum(content_risk_factors) / len(content_risk_factors) * 0.3
    else:
        content_score = 0.0

    total_score = url_score + content_score

    # 3. Trusted domain override - STRONGER EFFECT
    if is_trusted_domain:
        total_score *= 0.3  # Strongly reduce score for trusted domains

    # Ensure minimum score for certain high-risk indicators
    if url_features[11] == 1:  # Invalid TLD
        total_score = max(total_score, 0.6)  # Reduced from 0.7

    if url_features[10] == 1:  # Protocol typos
        total_score = max(total_score, 0.5)  # Reduced from 0.6

    if url_features[14] == 1:  # Suspicious path
        total_score = max(total_score, 0.7)

    if len(lexical_flags) >= 3:  # Increased threshold
        total_score = max(total_score, 0.5)  # Reduced from 0.6

    return total_score, "PHISHING" if total_score > 0.6 else "SUSPICIOUS" if total_score > 0.3 else "SAFE", total_score

# ========== ENHANCED PREDICTION FUNCTION - FIXED ==========
def predict_url_single(url):
    """
    Enhanced prediction optimized for dataset performance - FIXED DECISION LOGIC
    """
    try:
        pipeline = GLOBAL_PIPELINE
        analysis_report = []
        screenshot_url = None
        website_content = ""
        suspicious_content_display = ""
        network_report = ""
        lexical_flags = []

        if pipeline['model'] is None:
            return " Please train model first using the 'Quick Train' tab first.", None, "Please train model first", "Please train model first", "Please train model first"

        print(f"Analyzing URL: {url}")

        # --- 0. BASIC URL VALIDATION ---
        analysis_report.append("## Comprehensive Security Analysis Report")
        analysis_report.append("")

        # Check if URL is empty or too short
        if not url or len(url.strip()) < 4:
            return "**ERROR**: Please enter a valid URL", None, "Invalid URL", "Invalid URL", "Invalid URL"

        # Basic URL format validation
        url_lower = url.lower().strip()
        if not re.match(r'^[a-zA-Z0-9]+://', url_lower):
            url_lower = "http://" + url_lower
            url = url_lower

        # --- 1. LEXICAL VALIDATION ---
        try:
            lexical_flags, is_trusted = validate_url_structure(url_lower)
            url_features = extract_url_features(url_lower)
            print(f"URL Features: {url_features}")
            print(f"Lexical Flags: {lexical_flags}")
        except Exception as e:
            lexical_flags = ["URL parsing error - possible malformed URL"]
            is_trusted = False
            url_features = [0] * 16

        # --- 2. NETWORK ANALYSIS ---
        try:
            network_info = analyze_network_information(url_lower)
            network_report = format_network_report(network_info)
        except Exception as e:
            network_report = f" Network Analysis Error: {str(e)}"
            network_info = {}

        # --- 3. WEBSITE CONTENT ANALYSIS ---
        try:
            website_text, content_flags, suspicious_text_found = extract_website_content(url_lower)

            # Format website content for display
            if website_text and len(website_text) > 0:
                website_content = website_text[:3000] + "..." if len(website_text) > 3000 else website_text
            else:
                website_content = "No text content could be extracted from the website."

            # Format suspicious content display
            if suspicious_text_found:
                suspicious_content_display = "**SUSPICIOUS CONTENT DETECTED:**\n\n"
                for i, text in enumerate(suspicious_text_found, 1):
                    suspicious_content_display += f"{i}. {text}\n\n"
            else:
                suspicious_content_display = " No suspicious text patterns detected in website content."

        except Exception as e:
            website_content = f"Error extracting website content: {str(e)}"
            suspicious_content_display = "Content analysis failed due to URL issues"
            content_flags = []
            suspicious_text_found = []

        # --- 4. ML MODEL PREDICTION ---
        ml_decision = "SAFE"
        ml_confidence = 0.0

        if pipeline['model'] is not None:
            try:
                X = pd.DataFrame([url_features], columns=ENHANCED_FEATURE_NAMES)
                X_scaled = pipeline['scaler'].transform(X) if pipeline['scaler'] else X

                print(f" Scaled features for ML: {X_scaled[0]}")

                # Handle both supervised and unsupervised models
                if hasattr(pipeline['model'], 'predict'):
                    if isinstance(pipeline['model'], IsolationForest):
                        # Unsupervised approach
                        prediction = pipeline['model'].predict(X_scaled)[0]
                        ml_decision = "PHISHING" if prediction == -1 else "SAFE"
                        ml_confidence = 0.9 if prediction == -1 else 0.1
                        print(f" IsolationForest prediction: {prediction} -> {ml_decision}")
                    else:
                        # Supervised models
                        prediction = pipeline['model'].predict(X_scaled)[0]
                        ml_decision = "PHISHING" if prediction == 1 else "SAFE"
                        if hasattr(pipeline['model'], "predict_proba"):
                            proba = pipeline['model'].predict_proba(X_scaled)[0]
                            ml_confidence = proba[1] if prediction == 1 else proba[0]
                            print(f" Supervised prediction: {prediction}, probabilities: {proba}")
                        else:
                            ml_confidence = 0.9 if prediction == 1 else 0.1
                            print(f"Supervised prediction: {prediction}")
            except Exception as e:
                print(f" ML prediction error: {e}")
                ml_decision = "ERROR"
                ml_confidence = 0.0

        # --- 5. FINAL DECISION - FIXED THRESHOLDS ---
        # Use ML model as primary decision maker with balanced approach
        if ml_decision == "PHISHING" and ml_confidence > 0.7:
            final_decision = "PHISHING: High Risk Detected"
            risk_color = "red"
            risk_level = "HIGH"
            final_score = 0.9
        else:
            # BALANCED rule-based scoring
            try:
                combined_score, combined_decision, confidence = calculate_combined_score(
                    url_features, content_flags, suspicious_text_found, is_trusted, lexical_flags
                )
                final_score = combined_score

                # PROPER THRESHOLDS FOR ALL THREE CATEGORIES
                if final_score >= 0.7:  # High threshold for phishing
                    final_decision = "PHISHING: High Risk Detected"
                    risk_color = "red"
                    risk_level = "HIGH"
                elif final_score >= 0.4:  # Medium threshold for suspicious
                    final_decision = "SUSPICIOUS: Medium Risk"
                    risk_color = "orange"
                    risk_level = "MEDIUM"
                elif is_trusted and final_score < 0.2:
                    final_decision = "SAFE: Trusted Website"
                    risk_color = "green"
                    risk_level = "LOW"
                else:
                    final_decision = "SAFE: Low Risk"
                    risk_color = "green"
                    risk_level = "LOW"

            except Exception as e:
                final_decision = "SAFE: Low Risk"
                risk_color = "green"
                risk_level = "LOW"
                final_score = 0.1

        # ========== TABLE FORMATTED REPORT ==========

        # Final Verdict Section
        analysis_report.append(f"###  Final Verdict")
        analysis_report.append(f"<div style='background-color: {risk_color}; color: white; padding: 15px; border-radius: 10px; text-align: center; font-weight: bold; font-size: 18px;'>")
        analysis_report.append(f"{final_decision}")
        analysis_report.append(f"</div>")
        analysis_report.append("")

        # Risk Score Section
        analysis_report.append("###  Risk Assessment Summary")
        analysis_report.append("| Metric | Value | Risk Level |")
        analysis_report.append("|--------|-------|------------|")

        # Overall Score
        score_color = "red" if final_score >= 0.7 else "orange" if final_score >= 0.4 else "green"
        analysis_report.append(f"| Overall Risk Score | {final_score:.1%} | <span style='color:{score_color}; font-weight:bold'>{risk_level}</span> |")

        # ML Model Result
        ml_color = "red" if ml_decision == "PHISHING" else "green"
        ml_risk = "HIGH" if ml_decision == "PHISHING" else "LOW"
        analysis_report.append(f"| ML Model | {ml_decision} | <span style='color:{ml_color}; font-weight:bold'>{ml_risk}</span> |")

        # URL Analysis
        url_risk = "HIGH" if len(lexical_flags) >= 3 else "MEDIUM" if len(lexical_flags) >= 1 else "LOW"
        url_color = "red" if len(lexical_flags) >= 3 else "orange" if len(lexical_flags) >= 1 else "green"
        analysis_report.append(f"| URL Structure | {len(lexical_flags)} issues | <span style='color:{url_color}; font-weight:bold'>{url_risk}</span> |")

        # Content Analysis
        content_risk = "HIGH" if len(content_flags) >= 3 else "MEDIUM" if len(content_flags) >= 1 else "LOW"
        content_color = "red" if len(content_flags) >= 3 else "orange" if len(content_flags) >= 1 else "green"
        analysis_report.append(f"| Content Analysis | {len(content_flags)} flags | <span style='color:{content_color}; font-weight:bold'>{content_risk}</span> |")

        # Trust Status
        trust_color = "green" if is_trusted else "orange"
        trust_status = "VERIFIED" if is_trusted else "UNVERIFIED"
        analysis_report.append(f"| Domain Trust | {trust_status} | <span style='color:{trust_color}; font-weight:bold'>{trust_status}</span> |")
        analysis_report.append("")

        # Detailed Analysis Section
        analysis_report.append("###  Detailed Analysis")

        # URL Structure Analysis
        analysis_report.append("#### URL Structure Analysis")
        if lexical_flags:
            analysis_report.append("<div style='background-color: #ffebee; padding: 10px; border-radius: 5px; border-left: 4px solid #f44336;'>")
            analysis_report.append("**Issues Found:**")
            for flag in lexical_flags:
                flag_color = "red" if any(keyword in flag.lower() for keyword in ["invalid tld", "suspicious protocol", "malformed"]) else "orange"
                analysis_report.append(f"- <span style='color:{flag_color}; font-weight:bold'>{flag}</span>")
            analysis_report.append("</div>")
        else:
            analysis_report.append("<div style='background-color: #e8f5e8; padding: 10px; border-radius: 5px; border-left: 4px solid #4caf50;'>")
            analysis_report.append("<span style='color:green; font-weight:bold'>No suspicious URL patterns detected</span>")
            analysis_report.append("</div>")
        analysis_report.append("")

        # ML Model Analysis
        analysis_report.append("#### ML Model Analysis")
        analysis_report.append("<div style='background-color: #e3f2fd; padding: 10px; border-radius: 5px; border-left: 4px solid #2196f3;color:black;'>")
        analysis_report.append(f"**Model Prediction:** <span style='color:{ml_color}; font-weight:bold'>{ml_decision}</span>")
        analysis_report.append(f"**Confidence:** {ml_confidence:.1%}")
        analysis_report.append("</div>")
        analysis_report.append("")

        # Content Analysis
        analysis_report.append("#### Content Analysis")
        if content_flags:
            analysis_report.append("<div style='background-color: #ffebee; padding: 10px; border-radius: 5px; border-left: 4px solid #f44336;color:black;'>")
            analysis_report.append("**Content Red Flags:**")
            for flag in content_flags:
                flag_color = "red" if any(keyword in flag.lower() for keyword in ["password", "external domain", "suspicious text"]) else "orange"
                analysis_report.append(f"- <span style='color:{flag_color}; font-weight:bold'>{flag}</span>")
            analysis_report.append("</div>")
        else:
            analysis_report.append("<div style='background-color: #e8f5e8; padding: 10px; border-radius: 5px; border-left: 4px solid #4caf50;'>")
            analysis_report.append("<span style='color:green; font-weight:bold'>No suspicious content patterns detected</span>")
            analysis_report.append("</div>")
        analysis_report.append("")

        # Recommendations
        analysis_report.append("###  Security Recommendations")
        if risk_level == "HIGH":
            analysis_report.append("<div style='background-color: #ffebee; padding: 15px; border-radius: 5px; border: 2px solid #f44336;color:black;'>")
            analysis_report.append("<span style='color:red; font-weight:bold'>CRITICAL WARNING:</span>")
            analysis_report.append("- Do not enter any personal information")
            analysis_report.append("- Do not download any files")
            analysis_report.append("- Close this website immediately")
            analysis_report.append("- This site exhibits multiple phishing characteristics")
            analysis_report.append("</div>")
        elif risk_level == "MEDIUM":
            analysis_report.append("<div style='background-color: #fff3e0; padding: 15px; border-radius: 5px; border: 2px solid #ff9800;color:black;'>")
            analysis_report.append("<span style='color:orange; font-weight:bold'>CAUTION ADVISED:</span>")
            analysis_report.append("- Be cautious with personal information")
            analysis_report.append("- Verify the website legitimacy through other means")
            analysis_report.append("- Look for SSL certificate and contact information")
            analysis_report.append("- This site shows suspicious characteristics")
            analysis_report.append("</div>")
        else:
            analysis_report.append("<div style='background-color: #e8f5e8; padding: 15px; border-radius: 5px; border: 2px solid #4caf50;color:black;'>")
            analysis_report.append("<span style='color:green; font-weight:bold'>SECURE:</span>")
            analysis_report.append("- Website appears safe for browsing")
            analysis_report.append("- Standard security precautions still recommended")
            analysis_report.append("- Always verify SSL certificate for sensitive transactions")
            analysis_report.append("</div>")

        # Try to get screenshot
        try:
            screenshot_url = get_screenshot_url(url_lower)
        except:
            screenshot_url = None

        return "\n".join(analysis_report), screenshot_url, website_content, suspicious_content_display, network_report

    except Exception as e:
        error_msg = f"## Critical Analysis Error\n\n**Error Details:** {str(e)}\n\nThis usually happens with extremely malformed URLs. Please check the URL format and try again."
        return error_msg, None, f"Error: {str(e)}", f"Error: {str(e)}", f"Error: {str(e)}"

# ========== QR CODE ANALYSIS FUNCTION ==========
# ========== FIXED QR CODE ANALYSIS FUNCTION ==========
def analyze_qr_code(image):
    """
    Complete QR code analysis pipeline with proper URL extraction
    """
    try:
        # Step 1: Scan QR code using advanced method
        success, url_result, annotated_image = scan_qr_code_advanced(image)
        
        if not success:
            return f"{url_result}", None, "No URL extracted", "No URL extracted", "No URL extracted"
        
        # Extract the actual URL from the result
        if "Multiple URLs found:" in url_result:
            urls = url_result.replace("Multiple URLs found: ", "").split(", ")
            extracted_url = urls[0]
            scan_message = f"QR Code Scanned Successfully!\n\nFound {len(urls)} URLs. Analyzing the first one:\n**{extracted_url}**\n\nOther URLs found: {', '.join(urls[1:])}"
        else:
            extracted_url = url_result
            scan_message = f" QR Code Scanned Successfully!\n\nExtracted URL:\n**{extracted_url}**"
        
        # Step 2: Handle QR redirection services
        final_url = extracted_url
        
        # List of known QR redirection services
        qr_redirect_services = [
            'q.me-qr.com', 'qr.li', 'qr.io', 'goqr.me', 'qrickit.com',
            'scan.me', 'beaconstac.com', 'qrcode-monkey.com', 'qr-code-generator.com',
            'qrstuff.com', 'qr-code.com', 'qrd.by', 'qrbarcode.com'
        ]
        
        redirect_detected = False
        redirect_service = None
        
        for service in qr_redirect_services:
            if service in extracted_url:
                redirect_detected = True
                redirect_service = service
                break
        
        # If it's a redirect service, try to resolve the final destination
        if redirect_detected:
            try:
                print(f"🔗 QR Redirect detected: {redirect_service}")
                print(f" Resolving redirect from: {extracted_url}")
                
                # Follow redirects to get final URL
                response = requests.get(
                    extracted_url, 
                    headers=REQUEST_HEADERS, 
                    timeout=10, 
                    allow_redirects=True
                )
                final_url = response.url
                
                print(f" Final destination: {final_url}")
                
                scan_message += f"\n\n**QR Redirect Service Detected:** {redirect_service}"
                scan_message += f"\n **Final Destination:** {final_url}"
                scan_message += f"\n **Note:** This QR uses a redirect service. Analyzing the final destination."
                
            except Exception as e:
                print(f"❌ Redirect resolution failed: {e}")
                scan_message += f"\n\n⚠️ **Warning:** Could not resolve QR redirect service. Analyzing the redirect URL itself."
        
        # Step 3: Analyze the final URL
        analysis_report, screenshot, website_content, suspicious_content, network_report = predict_url_single(final_url)
        
        # Add QR-specific information to the report
        qr_enhanced_report = f"## QR Code Security Analysis\n\n**QR Scan Result:** {scan_message}\n\n**Analyzed URL:** {final_url}\n\n{analysis_report}"
        
        return qr_enhanced_report, screenshot, website_content, suspicious_content, network_report
        
    except Exception as e:
        error_msg = f"## QR Code Analysis Error\n\n**Error Details:** {str(e)}\n\nPlease ensure the QR code image is clear and contains a valid URL."
        return error_msg, None, f"Error: {str(e)}", f"Error: {str(e)}", f"Error: {str(e)}"

# ========== QR CODE GENERATION FUNCTION ==========
def generate_qr_code(url):
    """
    Generate a QR code for a given URL
    Returns: PIL Image of the QR code
    """
    try:
        qr = qrcode.QRCode(
            version=1,
            error_correction=qrcode.constants.ERROR_CORRECT_L,
            box_size=10,
            border=4,
        )
        qr.add_data(url)
        qr.make(fit=True)
        
        qr_img = qr.make_image(fill_color="black", back_color="white")
        
        # Add URL text below QR code
        img_with_text = Image.new('RGB', (qr_img.size[0], qr_img.size[1] + 40), 'white')
        img_with_text.paste(qr_img, (0, 0))
        
        draw = ImageDraw.Draw(img_with_text)
        try:
            font = ImageFont.truetype("arial.ttf", 12)
        except:
            font = ImageFont.load_default()
        
        # Truncate long URLs for display
        display_url = url[:50] + "..." if len(url) > 50 else url
        text_width = draw.textlength(display_url, font=font)
        text_x = (qr_img.size[0] - text_width) // 2
        draw.text((text_x, qr_img.size[1] + 10), display_url, fill="black", font=font)
        
        return img_with_text
        
    except Exception as e:
        # Return error image
        error_img = Image.new('RGB', (200, 200), 'white')
        draw = ImageDraw.Draw(error_img)
        draw.text((10, 90), f"Error: {str(e)}", fill="red")
        return error_img

# ========== GRADIO UI ==========
custom_theme = gr.themes.Soft(
    primary_hue="purple",
    secondary_hue="purple"
).set(
    body_background_fill="linear-gradient(135deg, #ffffff 0%, #f8f7ff 50%, #f0edff 100%)",
    background_fill_primary="#ffffff",
    background_fill_secondary="#f5f3ff",
    border_color_primary="#e2dbfc",
    color_accent_soft="#8b5cf6",
)

with gr.Blocks(theme=custom_theme) as demo:
    gr.Markdown("""
    <style>
    .purple-center {
        text-align: center;
        color: #800080;
        font-weight: bold;
    }
    </style>

    <div class="purple-center">
        <h1>PHISHING URL DETECTOR</h1>
        <p>Hybrid analysis with Lexical features, Live Web Crawler & ML Models</p>
        <p><strong>NEW:</strong> QR Code Scanning for Quishing Detection</p>
    </div>
    """)

    with gr.Tabs():
        with gr.TabItem(" Quick Train"):
            gr.Markdown("### Train your phishing detection model")

            # Model selection bar at top
            with gr.Row():
                model_sel = gr.Dropdown(
                    choices=["DecisionTree", "KNN", "IsolationForest"],
                    value="DecisionTree",
                    label="Select Model Type"
                )
                train_btn = gr.Button(" Start Training", variant="primary", size="lg")

            # Training summary below model selection
            with gr.Row():
                summary_box = gr.Markdown(
                    label="Training Results",
                    value="**Welcome!** \n\nClick 'Start Training' to train the optimized phishing detector. The system will automatically:\n\n• Load your dataset from Google Colab\n• Handle all three label types (benign, phishing, defacement)\n• Extract enhanced features optimized for your data\n• Test the model on dataset samples for verification\n\nAfter training, use the 'Check URL' tab to analyze any website."
                )

            # Confusion Matrix and Training Statistics side by side
            with gr.Row():
                with gr.Column(scale=1):
                    cm_image = gr.Plot(label="Confusion Matrix")
                with gr.Column(scale=1):
                    stats_image = gr.Plot(label="Training Statistics")

            # Additional info and download
            with gr.Row():
                info_box = gr.Textbox(
                    label="Status",
                    value="Ready to train! System will auto-detect your dataset and optimize for performance.",
                    lines=2
                )

            download_model = gr.File(label=" Download Trained Model", visible=True)

            train_btn.click(
                fn=train_pipeline,
                inputs=[model_sel],
                outputs=[summary_box, cm_image, stats_image, info_box]
            ).then(
                lambda: "trained_pipeline.joblib",
                outputs=[download_model]
            )

        with gr.TabItem(" Check URL (with Crawler)"):
            gr.Markdown("### Test any URL")

            # URL input and examples side by side
            with gr.Row():
                with gr.Column(scale=1):
                    url_input = gr.Textbox(
                        label="🔗 Enter URL to analyze",
                        placeholder="https://example.com",
                        value="https://en.wikipedia.org",
                        lines=2
                    )
                    predict_btn = gr.Button(" Analyze URL", variant="primary", size="lg")

                with gr.Column(scale=1):
                    gr.Markdown("### Test these examples:")
                    gr.Examples(
                        examples=[
                            "https://google.com",
                            "https://github.com",
                            "http://web.tv/liveCategory/17/language/1/index/changeLanguage/newshared/searchAutoComplete",
                            "http://goooooooogle.com",
                            "http://www.brasilcielo.comeze.com/"
                            
                        ],
                        inputs=url_input,
                        label="Dataset-Optimized Test Cases"
                    )

            # Security Analysis and Website Preview
            with gr.Row():
                prediction_output = gr.Markdown(
                    label="Security Analysis",
                    value="** Enhanced Hybrid Detection Active** \n\n1. **Lexical Analysis** (URL patterns)\n2. **Live Web Crawler** (Page content)\n3. **ML Model** (Dataset-optimized verification)\n4. **Network Analysis** (Security checks)\n\n✅ Ready to analyze. Please train the model first for optimal performance."
                )

            # Screenshot and Network Details side by side (same size)
            with gr.Row(equal_height=True):
                with gr.Column(scale=1):
                    screenshot_output = gr.Image(
                        label=" Website Preview",
                        value=None,
                        height=300
                    )
                with gr.Column(scale=1):
                    network_analysis_output = gr.Textbox(
                        label=" Network Security Details",
                        lines=8,
                        show_copy_button=True
                    )

            # Website Content and Suspicious Content side by side (same size)
            with gr.Row(equal_height=True):
                with gr.Column(scale=1):
                    website_content_output = gr.Textbox(
                        label=" Website Content (Extracted Text)",
                        placeholder="Website content will appear here for EVERY analysis...",
                        lines=8,
                        show_copy_button=True,
                        max_lines=10
                    )
                with gr.Column(scale=1):
                    suspicious_content_output = gr.Textbox(
                        label=" Suspicious Content Detected",
                        placeholder="Suspicious text patterns will appear here...",
                        lines=8,
                        show_copy_button=True,
                        max_lines=10
                    )

            predict_btn.click(
                fn=predict_url_single,
                inputs=[url_input],
                outputs=[prediction_output, screenshot_output, website_content_output, suspicious_content_output, network_analysis_output]
            )

        with gr.TabItem(" QR Code Scanner (Quishing Detection)"):
            gr.Markdown("""
            ## 🔍 QR Code Security Scanner
            **Detect Quishing Attacks** - Scan QR codes to extract and analyze URLs for phishing threats
            
            
            
            
            """)

            with gr.Row():
                with gr.Column(scale=1):
                    qr_upload = gr.Image(
                        label="Upload QR Code Image",
                        type="pil",
                        sources=["upload"],
                        height=300
                    )
                    
                    with gr.Row():
                        qr_scan_btn = gr.Button(" Scan & Analyze QR Code", variant="primary", size="lg")
                        clear_qr_btn = gr.Button(" Clear", variant="secondary")
                    
                   

                with gr.Column(scale=1):
                    qr_analysis_output = gr.Markdown(
                        label="QR Code Security Analysis",
                        value="### Ready to Scan QR Codes\n\nUpload a QR code image to analyze it for quishing threats.\n\n**Supported formats:** PNG, JPG, JPEG\n**Detection methods:** URL extraction + full phishing analysis"
                    )

            # QR Analysis Results (same layout as URL analysis)
            with gr.Row(equal_height=True):
                with gr.Column(scale=1):
                    qr_screenshot_output = gr.Image(
                        label=" Website Preview",
                        value=None,
                        height=300
                    )
                with gr.Column(scale=1):
                    qr_network_output = gr.Textbox(
                        label=" Network Security Details",
                        lines=8,
                        show_copy_button=True
                    )

            with gr.Row(equal_height=True):
                with gr.Column(scale=1):
                    qr_website_content = gr.Textbox(
                        label=" Website Content (Extracted Text)",
                        lines=8,
                        show_copy_button=True,
                        max_lines=10
                    )
                with gr.Column(scale=1):
                    qr_suspicious_content = gr.Textbox(
                        label=" Suspicious Content Detected",
                        lines=8,
                        show_copy_button=True,
                        max_lines=10
                    )

            

            # QR scanning functionality
            qr_scan_btn.click(
                fn=analyze_qr_code,
                inputs=[qr_upload],
                outputs=[qr_analysis_output, qr_screenshot_output, qr_website_content, qr_suspicious_content, qr_network_output]
            )
          
            
            # Clear QR inputs
                        # Clear QR inputs
            clear_qr_btn.click(
                lambda: [None, None, None, None, None,None],
                outputs=[qr_upload, qr_analysis_output, qr_screenshot_output, qr_website_content, qr_suspicious_content, qr_network_output]
            )

print("Launching enhanced phishing detector - DATASET OPTIMIZED - FIXED - WITH QR SCANNING...")
print("Looking for dataset files in Google Colab...")
demo.launch(share=True, debug=False, show_error=True)
