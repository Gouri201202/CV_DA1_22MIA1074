import streamlit as st
import cv2
import numpy as np
from PIL import Image
import pandas as pd


# ============================================================================
# PAGE CONFIGURATION
# ============================================================================
st.set_page_config(
    page_title="Shape & Contour Analyzer",
    page_icon="🔍",
    layout="wide"
)


st.title("🔍 Shape & Contour Analyzer")
st.markdown("""
**Interactive Computer Vision Dashboard** - Upload an image to detect geometric shapes, 
count objects, and measure contour features like area, perimeter, circularity, and solidity.
""")


# ============================================================================
# SIDEBAR CONTROLS
# ============================================================================
st.sidebar.header("⚙️ Processing Parameters")


min_area = st.sidebar.slider(
    "Minimum contour area (pixels²)",
    min_value=100,
    max_value=5000,
    value=800,
    step=100,
    help="Filter out small noise by setting minimum area threshold"
)


epsilon_factor = st.sidebar.slider(
    "Shape approximation factor",
    min_value=0.01,
    max_value=0.10,
    value=0.03,
    step=0.01,
    help="Lower values = more vertices, Higher values = smoother polygons"
)


show_features = st.sidebar.checkbox(
    "Show advanced features",
    value=True,
    help="Display circularity, solidity, extent in results table"
)


st.sidebar.markdown("---")
st.sidebar.markdown("""
###  Outcomes
- **Contour detection** with OpenCV
- **Feature extraction**: area, perimeter, vertices
- **Shape classification** using geometry
- **Morphological operations** for preprocessing
""")

# Current settings display
st.sidebar.markdown("---")
st.sidebar.markdown("### ⚙️ Current Settings")
st.sidebar.info(f"""
**Min Area:** {min_area} px²  
**Epsilon:** {epsilon_factor}  
**Features:** {'Shown' if show_features else 'Hidden'}
""")


# ============================================================================
# SHAPE CLASSIFIER (YOUR WORKING VERSION)
# ============================================================================
def classify_shape_ultimate(cnt, epsilon_factor=0.03):
    """Advanced shape classifier with multiple contour features."""
    
    area = cv2.contourArea(cnt)
    peri = cv2.arcLength(cnt, True)
    if peri == 0 or area < 100:
        return "Noise", None, {}
    
    # Core geometric features
    circularity = 4 * np.pi * area / (peri ** 2)
    hull = cv2.convexHull(cnt)
    hull_area = cv2.contourArea(hull)
    solidity = area / hull_area if hull_area > 0 else 0
    
    x, y, w, h = cv2.boundingRect(cnt)
    aspect_ratio = w / h if h > 0 else 0
    extent = area / (w * h) if (w * h) > 0 else 0
    
    epsilon = epsilon_factor * peri
    approx = cv2.approxPolyDP(cnt, epsilon, True)
    vertices = len(approx)
    
    # Feature dictionary for display
    features = {
        'circularity': circularity,
        'solidity': solidity,
        'aspect_ratio': aspect_ratio,
        'extent': extent,
        'vertices': vertices
    }
    
    # === CLASSIFICATION LOGIC ===
    
    # HIGH-SOLIDITY SHAPES (>0.95)
    if solidity > 0.95:
        if vertices == 3:
            return "Triangle", approx, features
        elif vertices == 4:
            shape = "Square" if 0.95 <= aspect_ratio <= 1.05 else "Rectangle"
            return shape, approx, features
        elif vertices == 5:
            return "Pentagon", approx, features
        elif vertices == 6:
            return "Hexagon", approx, features
        elif circularity > 0.80 and vertices >= 8:
            if 0.9 <= aspect_ratio <= 1.1:
                return "Circle", approx, features
            else:
                return "Oval", approx, features
    
    # SEMI-CIRCLE
    elif vertices == 7 and circularity > 0.65 and solidity > 0.90:
        return "Semi-circle", approx, features
    
    # PARALLELOGRAM/TRAPEZOID
    elif 4 <= vertices <= 7 and 0.40 < circularity < 0.70 and solidity < 0.85:
        if vertices == 4:
            if 0.85 <= aspect_ratio <= 1.15:
                return "Parallelogram", approx, features
            else:
                return "Trapezoid", approx, features
        else:
            return "Irregular Quad", approx, features
    
    # HEART/ARROW
    elif 8 <= vertices <= 12 and 0.50 < circularity < 0.75 and 0.75 < solidity < 0.90:
        return "Heart/Arrow", approx, features
    
    # CROSS
    elif circularity < 0.50 and extent < 0.60 and 4 <= vertices <= 8:
        return "Cross/Arrow", approx, features
    
    # STAR
    elif vertices >= 10 and circularity < 0.50:
        return "Star", approx, features
    
    # OVAL/ELLIPSE
    elif circularity > 0.80 and (aspect_ratio < 0.8 or aspect_ratio > 1.2):
        return "Oval", approx, features
    
    # DEFAULT
    return f"Polygon-{vertices}v", approx, features



# ============================================================================
# IMAGE PROCESSING FUNCTION
# ============================================================================
def process_image(image, min_area, epsilon_factor):
    """Process uploaded image and detect shapes."""
    
    # Convert PIL to numpy array
    img = np.array(image.convert('RGB'))
    original = img.copy()
    
    # Preprocessing pipeline
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Adaptive thresholding
    thresh = cv2.adaptiveThreshold(
        blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY_INV, 11, 2
    )
    
    # Morphological closing to fill gaps
    kernel = np.ones((3, 3), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    
    # Find contours
    contours, _ = cv2.findContours(
        thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    
    # Analyze each contour
    drawn = original.copy()
    results = []
    object_count = 0
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue
        
        peri = cv2.arcLength(cnt, True)
        shape_name, approx, features = classify_shape_ultimate(cnt, epsilon_factor)
        
        if shape_name == "Noise":
            continue
        
        object_count += 1
        
        # Draw contour (green)
        cv2.drawContours(drawn, [approx], -1, (0, 255, 0), 3)
        
        # Calculate centroid and add label (blue text)
        M = cv2.moments(cnt)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            
            # Add shape label
            cv2.putText(
                drawn, shape_name, (cX - 60, cY - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2
            )
        
        # Store results
        result_dict = {
            'ID': object_count,
            'Shape': shape_name,
            'Area (px²)': round(area, 1),
            'Perimeter (px)': round(peri, 1),
            'Vertices': features['vertices']
        }
        
        if show_features:
            result_dict.update({
                'Circularity': round(features['circularity'], 3),
                'Solidity': round(features['solidity'], 3),
                'Aspect Ratio': round(features['aspect_ratio'], 2)
            })
        
        results.append(result_dict)
    
    return original, thresh, drawn, results, object_count



# ============================================================================
# FILE UPLOADER
# ============================================================================
uploaded_file = st.file_uploader(
    "📁 Upload an image with geometric shapes",
    type=["png", "jpg", "jpeg"],
    help="Best results with clear shapes on contrasting backgrounds"
)


# ============================================================================
# MAIN APP LOGIC
# ============================================================================
if uploaded_file is not None:
    
    # Load image
    image = Image.open(uploaded_file)
    
    # Process image
    with st.spinner("🔄 Analyzing shapes..."):
        original, thresh, drawn, results, obj_count = process_image(
            image, min_area, epsilon_factor
        )
    
    # Display images in 3 columns
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader(" Original Image")
        st.image(original, use_container_width=True)
    
    with col2:
        st.subheader(" Binary Threshold")
        st.image(thresh, use_container_width=True, channels="GRAY")
    
    with col3:
        st.subheader(" Detected Shapes")
        st.image(drawn, use_container_width=True)
    
    # Display metrics
    st.markdown("---")
    metric_col1, metric_col2, metric_col3 = st.columns(3)
    
    with metric_col1:
        st.metric(" Total Objects Detected", obj_count)
    
    with metric_col2:
        if results:
            unique_shapes = len(set([r['Shape'] for r in results]))
            st.metric("Unique Shape Types", unique_shapes)
    
    with metric_col3:
        if results:
            total_area = sum([r['Area (px²)'] for r in results])
            st.metric(" Total Area", f"{total_area:.0f} px²")
    
    # Display results table
    if results:
        st.markdown("---")
        st.subheader(" Detected Object Details")
        
        df = pd.DataFrame(results)
        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True
        )
        
        # ========== LIVE SHAPE STATISTICS (NEW) ==========
        st.markdown("---")
        st.subheader("📈 Live Shape Statistics")
        
        # Shape distribution chart
        shape_counts = df['Shape'].value_counts()
        st.bar_chart(shape_counts)
        
        # Summary metrics
        summary_col1, summary_col2, summary_col3, summary_col4 = st.columns(4)
        
        with summary_col1:
            most_common = shape_counts.idxmax()
            st.metric("Most Common Shape", most_common, 
                     f"{shape_counts[most_common]} found")
        
        with summary_col2:
            avg_area = df['Area (px²)'].mean()
            st.metric("Average Area", f"{avg_area:.0f} px²")
        
        with summary_col3:
            avg_peri = df['Perimeter (px)'].mean()
            st.metric("Average Perimeter", f"{avg_peri:.0f} px")
        
        with summary_col4:
            avg_vertices = df['Vertices'].mean()
            st.metric("Avg. Vertices", f"{avg_vertices:.1f}")
        
        # Additional statistics in expander
        with st.expander("📊 Detailed Statistics"):
            stat_col1, stat_col2 = st.columns(2)
            
            with stat_col1:
                st.markdown("#### Area Statistics")
                st.write(f"**Min Area:** {df['Area (px²)'].min():.1f} px²")
                st.write(f"**Max Area:** {df['Area (px²)'].max():.1f} px²")
                st.write(f"**Median Area:** {df['Area (px²)'].median():.1f} px²")
                st.write(f"**Std Dev:** {df['Area (px²)'].std():.1f} px²")
            
            with stat_col2:
                st.markdown("#### Perimeter Statistics")
                st.write(f"**Min Perimeter:** {df['Perimeter (px)'].min():.1f} px")
                st.write(f"**Max Perimeter:** {df['Perimeter (px)'].max():.1f} px")
                st.write(f"**Median Perimeter:** {df['Perimeter (px)'].median():.1f} px")
                st.write(f"**Std Dev:** {df['Perimeter (px)'].std():.1f} px")
            
            st.markdown("#### Shape Distribution")
            shape_dist_df = pd.DataFrame({
                'Shape': shape_counts.index,
                'Count': shape_counts.values,
                'Percentage': (shape_counts.values / shape_counts.sum() * 100).round(1)
            })
            st.dataframe(shape_dist_df, use_container_width=True, hide_index=True)
        
        # Download CSV option
        st.markdown("---")
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Results as CSV",
            data=csv,
            file_name="shape_analysis_results.csv",
            mime="text/csv"
        )
    
    # Explanation expander
    st.markdown("---")
    with st.expander(" How does this work?"):
        st.markdown("""
        ### Image Processing Pipeline
        
        1. **Grayscale Conversion**: Convert RGB to grayscale for simpler processing
        2. **Gaussian Blur**: Reduce noise with 5×5 kernel
        3. **Adaptive Thresholding**: Create binary image with local threshold values
        4. **Morphological Closing**: Fill small gaps in contours
        5. **Contour Detection**: Use `cv2.findContours` to extract object boundaries
        
        ### Feature Extraction
        
        For each detected contour, we compute:
        - **Area**: `cv2.contourArea()` - pixels inside the contour
        - **Perimeter**: `cv2.arcLength()` - boundary length
        - **Circularity**: `4π×Area/Perimeter²` - measures roundness (1.0 = perfect circle)
        - **Solidity**: `Area/ConvexHullArea` - detects concave shapes (< 1.0)
        - **Vertices**: Polygon approximation with `cv2.approxPolyDP()`
        
        ### Shape Classification
        
        Uses hierarchical decision tree based on:
        - Number of vertices (3=triangle, 4=quad, 5=pentagon, etc.)
        - Aspect ratio (square vs rectangle)
        - Circularity (circle vs polygon)
        - Solidity (heart/star detection for concave shapes)
        """)


else:
    # ========== ENHANCED HOME PAGE (NEW) ==========
    st.info(" **Upload an image** to start analyzing shapes!")
    
    st.markdown("---")
    
    # Main features section
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("###  What This App Does")
        st.markdown("""
        - **Detects** geometric shapes (circles, triangles, squares, etc.)
        - **Counts** total objects in the image
        - **Measures** area, perimeter, and contour properties
        - **Classifies** complex shapes (hearts, stars, arrows, etc.)
        - **Exports** results as downloadable CSV
        - **Visualizes** shape distribution with live statistics
        """)
        
        st.markdown("###  Detected Shape Types")
        st.markdown("""
        **Basic Shapes:** Triangle, Square, Rectangle, Pentagon, Hexagon
        
        **Curved Shapes:** Circle, Oval, Semi-circle
        
        **Complex Shapes:** Star, Heart, Arrow, Cross, Parallelogram, Trapezoid
        """)
    
    with col2:
        st.markdown("###  Tips for Best Results")
        st.markdown("""
         Use images with **clear, distinct shapes**
        
         **High contrast** backgrounds work best (white/black)
        
         Avoid **overlapping shapes** for accurate counting
        
         Adjust **minimum area** slider to filter noise
        
         Try different **approximation values** for complex shapes
        
         Supported formats: **PNG, JPG, JPEG**
        """)
        
        st.markdown("### Features Analyzed")
        st.markdown("""
        - **Area** - Size in square pixels
        - **Perimeter** - Boundary length in pixels
        - **Circularity** - Measures roundness (1.0 = perfect circle)
        - **Solidity** - Detects concave/convex shapes
        - **Vertices** - Number of corner points
        - **Aspect Ratio** - Width to height ratio
        """)
    
    st.markdown("---")
    
    # Sample images section
    st.markdown("###  Sample Test Images")
    st.markdown("""
    For testing, use images with:
    - Geometric shape collections
    - Educational diagrams
    - Logo designs with shapes
    - Icon sets
    - Pattern recognition exercises
    """)
    
    # Technical details in expander
    with st.expander(" Technical Details - How It Works"):
        st.markdown("""
        ### Computer Vision Pipeline
        
        **1. Image Preprocessing**
        ```
        RGB → Grayscale → Gaussian Blur → Adaptive Thresholding
        ```
        
        **2. Contour Detection**
        - Uses OpenCV's `findContours()` with `RETR_EXTERNAL` mode
        - Extracts object boundaries from binary image
        - Filters by minimum area threshold
        
        **3. Feature Extraction**
        
        | Feature | Formula | Purpose |
        |---------|---------|---------|
        | Area | `cv2.contourArea()` | Object size |
        | Perimeter | `cv2.arcLength()` | Boundary length |
        | Circularity | `4π × Area / Perimeter²` | Roundness measure |
        | Solidity | `Area / ConvexHullArea` | Concavity detection |
        | Vertices | `len(approxPolyDP())` | Corner points |
        
        **4. Shape Classification**
        - Polygon approximation using Douglas-Peucker algorithm
        - Multi-feature decision tree (vertices + circularity + solidity)
        - Distinguishes 15+ different shape types
        
        **5. Visualization**
        - Green contours drawn on original image
        - Blue labels with shape names
        - Centroid calculation using image moments
        """)
        
        st.markdown("###  Key Algorithms")
        st.markdown("""
        - **Adaptive Thresholding**: Handles varying lighting conditions
        - **Morphological Closing**: Fills small gaps in contours
        - **Convex Hull**: Detects concave shapes (stars, hearts)
        - **Polygon Approximation**: Reduces contour complexity
        - **Moment Calculation**: Finds shape centroids
        """)
    
    # Quick stats/info
    st.markdown("---")
    info_col1, info_col2, info_col3 = st.columns(3)
    
    with info_col1:
        st.markdown("###  Fast Processing")
        st.markdown("Real-time analysis of uploaded images")
    
    with info_col2:
        st.markdown("### Accurate Detection")
        st.markdown("Advanced multi-feature classifier")
    
    with info_col3:
        st.markdown("### Export Results")
        st.markdown("Download analysis as CSV file")


# ============================================================================
# FOOTER
# ============================================================================
st.markdown("---")

footer_col1, footer_col2, footer_col3 = st.columns(3)

with footer_col1:
    st.markdown("###  About")
    st.markdown("""
    Interactive computer vision dashboard for 
    geometric shape detection and analysis.
    """)

with footer_col2:
    st.markdown("###  Built With")
    st.markdown("""
    - Python 3.10+
    - OpenCV 4.9
    - Streamlit 1.31
    - NumPy & Pandas
    """)

with footer_col3:
    st.markdown("###  Version")
    st.markdown("""
    **v1.0** (Jan 2026)
    
    Shape & Contour Analyzer
    """)

st.markdown(
    "<p style='text-align: center; color: gray; margin-top: 2rem;'>"
    "Shape & Contour Analyzer © 2026 | Powered by OpenCV + Streamlit</p>",
    unsafe_allow_html=True
)
