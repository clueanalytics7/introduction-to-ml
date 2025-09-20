# app_classifier.py
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Set the page title and icon
st.set_page_config(page_title="Classifier Playground", page_icon="🧪", layout="wide")

# App title and description
st.title("The Ultimate Classifier Playground")
st.markdown("""
**Welcome to the playground!** 👋  
Here you can explore how different Machine Learning classifiers work.
* **Choose a dataset** and **select which features to use for training**.
* **Pick any two features** to visualize the decision boundary in 2D.
* **Pick a classifier** and **adjust its parameters** using the sliders.
* Watch the **decision boundary** and **performance metrics** update in real-time!
""")

# Load datasets (using a helper function with caching)
@st.cache_data
def load_dataset(name):
    """
    Loads the selected sklearn dataset.
    Caching prevents reloading on every interaction, making the app faster.
    """
    data_functions = {
        'Iris': datasets.load_iris,
        'Wine': datasets.load_wine,
        'Breast Cancer': datasets.load_breast_cancer
    }
    dataset = data_functions[name]()
    X = dataset.data
    y = dataset.target
    feature_names = dataset.feature_names
    target_names = dataset.target_names
    return X, y, feature_names, target_names

# Create a custom function to plot decision boundary for 2D visualization
def plot_decision_boundary_2d(clf, X_vis, y_vis, feature_names, target_names, ax):
    """
    Plots decision boundary for a classifier using two features for visualization.
    This function creates a surrogate classifier trained only on the two visualization features.
    """
    # Create a surrogate classifier of the same type, trained only on the 2D visualization data
    surrogate_clf = type(clf)(**clf.get_params())
    
    # Scale the visualization features
    scaler_vis = StandardScaler()
    X_vis_scaled = scaler_vis.fit_transform(X_vis)
    
    # Train the surrogate classifier on just the two visualization features
    surrogate_clf.fit(X_vis_scaled, y_vis)
    
    # Create mesh grid for plotting
    h = 0.02  # step size in the mesh
    x_min, x_max = X_vis_scaled[:, 0].min() - 0.5, X_vis_scaled[:, 0].max() + 0.5
    y_min, y_max = X_vis_scaled[:, 1].min() - 0.5, X_vis_scaled[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    
    # Predict using the surrogate classifier
    Z = surrogate_clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    # Plot decision boundary
    cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'][:len(target_names)])
    ax.contourf(xx, yy, Z, alpha=0.3, cmap=cmap_light)
    
    # Plot the data points
    cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'][:len(target_names)])
    scatter = ax.scatter(X_vis_scaled[:, 0], X_vis_scaled[:, 1], c=y_vis, 
                        cmap=cmap_bold, edgecolor='k', s=50, alpha=0.7)
    
    ax.set_xlabel(f"{feature_names[0]} (standardized)")
    ax.set_ylabel(f"{feature_names[1]} (standardized)")
    
    # Create legend
    legend_elements = scatter.legend_elements()[0]
    ax.legend(legend_elements, target_names, title="Classes", loc="best")
    
    return ax

# Sidebar for user inputs
st.sidebar.header("Controls")

# 1. Dataset Selector
dataset_name = st.sidebar.selectbox(
    "Select Dataset",
    ("Iris", "Wine", "Breast Cancer")
)
X, y, feature_names, target_names = load_dataset(dataset_name)

# Create a DataFrame for better display
df = pd.DataFrame(X, columns=feature_names)
df['Target'] = y

# 2. Feature Selection for TRAINING - Using expander to avoid clutter
with st.sidebar.expander("📊 Select Training Features", expanded=False):
    st.markdown("**Select which features to use to train the model:**")
    
    # Use multiselect for training features with a compact view
    training_feature_indices = st.multiselect(
        "Features for Training Model",
        options=list(range(len(feature_names))),
        format_func=lambda x: feature_names[x],
        default=list(range(min(4, len(feature_names)))),
        label_visibility="collapsed"
    )

# Ensure at least one feature is selected
if not training_feature_indices:
    st.sidebar.error("Please select at least one feature for training.")
    st.stop()

# Extract the selected training features
X_train_full = X[:, training_feature_indices]
selected_training_features = [feature_names[i] for i in training_feature_indices]

# Display selected training features in a compact way
if selected_training_features:
    st.sidebar.markdown(f"**Selected for training:**")
    if len(selected_training_features) <= 3:
        st.sidebar.write(", ".join(selected_training_features))
    else:
        st.sidebar.write(f"{len(selected_training_features)} features selected")
        with st.sidebar.expander("View selected features"):
            for feat in selected_training_features:
                st.write(f"• {feat}")

# 3. Feature Selection for VISUALIZATION
st.sidebar.subheader("Visualization Settings")
st.sidebar.markdown("**Select any 2 features to plot the decision boundary:**")

# Use selectboxes for visualization features
viz_feature_x = st.sidebar.selectbox(
    "X-Axis Feature",
    list(range(len(feature_names))),
    index=0,
    format_func=lambda x: feature_names[x]
)
viz_feature_y = st.sidebar.selectbox(
    "Y-Axis Feature",
    list(range(len(feature_names))),
    index=1,
    format_func=lambda x: feature_names[x]
)

# Extract the two selected features for visualization
X_viz = X[:, [viz_feature_x, viz_feature_y]]
viz_feature_names = [feature_names[viz_feature_x], feature_names[viz_feature_y]]

# 4. Classifier Selector and Parameter Sliders
st.sidebar.subheader("Classifier Settings")
classifier_name = st.sidebar.selectbox(
    "Choose Classifier",
    ("Logistic Regression", "k-Nearest Neighbors (k-NN)", "Decision Tree", "Support Vector Machine (SVM)")
)

# Define parameters for each classifier
params = {}
if classifier_name == "Logistic Regression":
    C = st.sidebar.slider("Inverse Regularization (C)", 0.01, 10.0, 1.0, help="Smaller values specify stronger regularization.")
    params['C'] = C
    clf = LogisticRegression(C=C, random_state=42, max_iter=1000)
elif classifier_name == "k-Nearest Neighbors (k-NN)":
    n_neighbors = st.sidebar.slider("Number of Neighbors (k)", 1, 20, 5, help="Number of neighbors to use for classification.")
    params['n_neighbors'] = n_neighbors
    clf = KNeighborsClassifier(n_neighbors=n_neighbors)
elif classifier_name == "Decision Tree":
    max_depth = st.sidebar.slider("Max Tree Depth", 1, 15, 3, help="Maximum depth of the tree. Controls model complexity.")
    params['max_depth'] = max_depth
    clf = DecisionTreeClassifier(max_depth=max_depth, random_state=42)
elif classifier_name == "Support Vector Machine (SVM)":
    C = st.sidebar.slider("Regularization (C)", 0.01, 10.0, 1.0, help="Penalty parameter C of the error term.")
    params['C'] = C
    clf = SVC(C=C, probability=True, random_state=42)

# 5. Train/Test Split Slider
test_size = st.sidebar.slider("Test Set Size (%)", 10, 40, 20, step=5) / 100

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X_train_full, y, test_size=test_size, random_state=42, stratify=y
)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train the classifier on ALL selected training features
clf.fit(X_train_scaled, y_train)

# Calculate accuracies
y_train_pred = clf.predict(X_train_scaled)
y_test_pred = clf.predict(X_test_scaled)
train_accuracy = accuracy_score(y_train, y_train_pred)
test_accuracy = accuracy_score(y_test, y_test_pred)

# MAIN VISUALIZATION AREA
st.subheader("Performance & Visualization")

# Create columns for layout
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("**Decision Boundary Visualization**")
    st.info(f"💡 Visualizing using: **{viz_feature_names[0]}** vs **{viz_feature_names[1]}**")
    st.info(f"🧠 Model trained using: {len(selected_training_features)} features")

    # Create the decision boundary plot using our custom function
    fig, ax = plt.subplots(figsize=(10, 8))
    
    plot_decision_boundary_2d(clf, X_viz, y, viz_feature_names, target_names, ax)
    ax.set_title(f"{classifier_name}\n2D Decision Boundary Projection\n(Trained on {len(selected_training_features)} features, visualized on 2)")
    
    st.pyplot(fig)

with col2:
    st.markdown("**Model Performance**")
    
    # Display accuracies
    st.metric("Training Accuracy", f"{train_accuracy:.2%}")
    st.metric("Test Accuracy", f"{test_accuracy:.2%}")
    
    # Display feature information in a compact way
    with st.expander(f"Training Features ({len(selected_training_features)})"):
        for i, feat in enumerate(selected_training_features):
            st.write(f"{i+1}. {feat}")
    
    # Display the chosen parameters
    st.markdown("**Classifier Parameters:**")
    for key, value in params.items():
        st.write(f"- `{key}`: {value}")
    
    # Generate and plot a learning curve
    st.markdown("**Learning Curve**")
    
    try:
        train_sizes, train_scores, test_scores = learning_curve(
            clf, X_train_scaled, y_train, cv=5, n_jobs=-1,
            train_sizes=np.linspace(0.1, 1.0, 5), random_state=42
        )
        
        train_scores_mean = np.mean(train_scores, axis=1)
        test_scores_mean = np.mean(test_scores, axis=1)
        
        fig_lc, ax_lc = plt.subplots(figsize=(6, 4))
        ax_lc.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
        ax_lc.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")
        ax_lc.set_xlabel("Training examples")
        ax_lc.set_ylabel("Score (Accuracy)")
        ax_lc.set_ylim(0.0, 1.1)
        ax_lc.grid()
        ax_lc.legend(loc="best")
        ax_lc.set_title("Model Learning Curve")
        
        st.pyplot(fig_lc)
    except Exception as e:
        st.warning("Could not generate learning curve for this configuration.")

# Add model interpretation
st.subheader("Model Analysis")
col_analysis1, col_analysis2 = st.columns(2)

with col_analysis1:
    st.markdown("**Performance Diagnosis:**")
    if train_accuracy is not None and test_accuracy is not None:
        gap = train_accuracy - test_accuracy
        if gap > 0.15:
            st.error("🚨 **Overfitting Detected:** Large gap between training and test performance. The model is too complex and memorizing the training data.")
            st.write("**Try:** Reducing model complexity (increase k for k-NN, decrease max depth for trees, decrease C for SVM/LR) or using more training data.")
        elif gap > 0.05:
            st.warning("⚠️ **Potential Overfitting:** Moderate gap between training and test performance.")
        elif train_accuracy < 0.7:
            st.info("📉 **Underfitting Detected:** Poor performance on both training and test data. The model is too simple.")
            st.write("**Try:** Increasing model complexity (decrease k for k-NN, increase max depth for trees, increase C for SVM/LR) or using more features.")
        else:
            st.success("✅ **Good Generalization:** Model performs consistently on both training and test data.")

with col_analysis2:
    st.markdown("**Dataset Overview:**")
    st.write(f"**Samples:** {X.shape[0]}")
    st.write(f"**Total Features:** {X.shape[1]}")
    st.write(f"**Features Used for Training:** {len(selected_training_features)}")
    st.write(f"**Classes:** {len(target_names)}")
    st.write(f"**Test Set Size:** {int(test_size * 100)}%")

# Show the raw data in an expandable section
with st.expander("Show Raw Dataset"):
    st.dataframe(df)

st.markdown("---")
st.caption("Experiment with different feature combinations and model parameters to see how they affect performance!")
