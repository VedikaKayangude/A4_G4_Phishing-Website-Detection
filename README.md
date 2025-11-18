# Phishing URL Detector

A comprehensive, multi-layered phishing detection system that combines lexical analysis, machine learning, live web crawling, and QR code scanning to identify malicious URLs and quishing attacks.

## Key Features

### **1. Multi-Layer Detection System**
- **Lexical URL Analysis**: Examines URL structure, patterns, and anomalies
- **Live Web Crawler**: Extracts and analyzes actual website content
- **Machine Learning Models**: Multiple algorithms trained on your dataset
- **Network Security Analysis**: DNS records, IP geolocation, and security checks
- **QR Code Scanning**: Detects quishing attacks via QR code analysis

### **2. Advanced QR Code Detection**
- **Multi-method QR scanning** using pyzbar and OpenCV
- **QR redirect service detection** (q.me-qr.com, qr.li, etc.)
- **Automatic redirection resolution** to analyze final destinations
- **Visual annotation** of detected QR codes with bounding boxes
- **QR code generation** for testing purposes

### **3. Multiple Machine Learning Models**
- **Decision Tree Classifier** (Recommended)
- **K-Nearest Neighbors (KNN)**
- **Isolation Forest** (Unsupervised)
- **Random Forest Classifier**
- Automatic dataset optimization and feature engineering

### **4. Comprehensive Security Analysis**
- **URL Structure Validation**: Protocol checks, TLD analysis, suspicious patterns
- **Content Analysis**: Form detection, suspicious text patterns, iframe analysis
- **Network Intelligence**: DNS record analysis, IP geolocation, ISP information
- **Real-time Risk Scoring**: Combined scoring from all detection layers

### **5. Enhanced Feature Extraction**
- 16 optimized features including:
  - URL length, dot count, hyphen count, digit count
  - Protocol validation, TLD analysis, keyword detection
  - Subdomain analysis, path analysis, character patterns
  - Trusted domain verification, parameter analysis

## Detection Capabilities

### **Phishing Indicators Detected**
- Suspicious keywords (login, verify, secure, bank, etc.)
- Protocol typos (htttp, htttps, httpss)
- Invalid/uncommon TLDs
- IP addresses in URLs
- Excessive URL encoding
- Character repetition patterns
- Suspicious path components
- External form submissions
- Password field detection
- Phishing-related text content

### **Risk Classification**
- **SAFE** (Low Risk): Trusted domains with no suspicious indicators
- **SUSPICIOUS** (Medium Risk): Some concerning patterns detected
- **PHISHING** (High Risk): Multiple confirmed phishing characteristics

## Dataset Support

### **Automatic Dataset Handling**
- Supports Excel (.xlsx) and CSV formats
- Auto-detects URL and label columns
- Handles multiple label formats:
  - Phishing: 'phishing', 'malicious', 'bad', '1'
  - Benign: 'benign', 'good', 'legitimate', '0'
  - Defacement: Treated as high-risk phishing

### **Optimized Feature Engineering**
- Enhanced feature extraction optimized for your dataset
- Automatic label standardization
- Balanced training with stratified sampling
- Comprehensive model testing on dataset samples

## User Interface

### **Three Main Tabs**

1. **Quick Train**
   - Model selection and training
   - Performance visualization
   - Confusion matrix and statistics
   - Model download capability

2. **Check URL (with Crawler)**
   - Real-time URL analysis
   - Website content extraction
   - Network security details
   - Visual website preview

3. **QR Code Scanner**
   - Upload and scan QR codes
   - Redirect service detection
   - Complete security analysis
   - Annotated QR code display

## 🔧 Technical Implementation

### **Dependencies**
```
Core: pandas, numpy, scikit-learn, gradio
Web: requests, beautifulsoup4, tldextract
Security: python-whois, dnspython, ssl
QR: opencv-python, pyzbar, qrcode, pillow
Visualization: matplotlib, plotly
```

### **System Requirements**
- Google Colab environment recommended
- libzbar0 system dependency for QR scanning
- Internet connection for live web crawling
- Sufficient RAM for dataset processing

##  Quick Start

1. **Upload your dataset** to Google Colab (phi.xlsx or similar)
2. **Launch the application** - system auto-detects your data
3. **Train the model** using the Quick Train tab
4. **Analyze URLs** or scan QR codes for security assessment

##  Performance Features

- **Hybrid Decision Making**: Combines ML predictions with rule-based scoring
- **Confidence Scoring**: Probability-based risk assessment
- **Real-time Analysis**: Live web content crawling
- **Comprehensive Reporting**: Detailed security analysis with recommendations
- **Dataset Optimization**: Automatically adapts to your specific data

## Security Recommendations

Based on risk level, the system provides:
- **HIGH RISK**: Immediate warnings with specific protective actions
- **MEDIUM RISK**: Caution advisories with verification steps
- **LOW RISK**: Security best practices for safe browsing

##  Use Cases

- **Enterprise Security**: Employee phishing awareness training
- **Financial Institutions**: Banking and payment protection
- **E-commerce**: Customer account security
- **Security Research**: Phishing campaign analysis
- **Personal Security**: Safe browsing and QR code verification



This enhanced phishing detector provides enterprise-grade security analysis with user-friendly interface, making advanced threat detection accessible to both security professionals and general users.
