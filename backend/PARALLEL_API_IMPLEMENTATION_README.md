# Parallel API Implementation: 2-Call Strategy

## Overview
This document provides a detailed implementation guide for splitting your current single LLM call into **2 parallel API calls**:
1. **Classification Call**: Fast activity type determination
2. **Extraction Call**: Comprehensive data extraction with RAG

## 🎯 **Current vs. Target Architecture**

### **Current System (Sequential)**
```
Input → Single LLM Call (GPT-4.1) → Full Response
├── Context: Full article + RAG + complete prompts
├── Task: Everything (classification + extraction)
├── Time: ~1.5 minutes
└── Cost: 1x GPT-4.1 call
```

### **Target System (Parallel)**
```
Input → Split Processing → Parallel Execution → Merge Results
├── Call 1: Classification (GPT-4o-mini)
│   ├── Context: Limited (2000 chars)
│   ├── Task: Activity type only
│   └── Time: 15-20 seconds
│
├── Call 2: Extraction (GPT-4.1)
│   ├── Context: Full article + RAG + complete prompts
│   ├── Task: All data fields
│   └── Time: 45-60 seconds
│
└── Total Time: max(classification, extraction) = 45-60 seconds
```

## 🚀 **Implementation Strategy**

### **Phase 1: Core Function Creation**

#### **1.1 Classification Function**
```python
async def classify_activity_type(input_text):
    """Fast classification call - determines activity type only"""
    try:
        classification_prompt = load_classification_prompt()
        
        # Create minimal context for classification
        classification_context = f"""
        ACTIVITY DESCRIPTION:
        {input_text[:2000]}  # Limit context for speed
        
        CLASSIFICATION TASK:
        {classification_prompt}
        """
        
        response = await llm_classification.ainvoke([
            {"role": "system", "content": classification_prompt},
            {"role": "user", "content": classification_context}
        ])
        
        # Extract activity type from response
        content = response.content
        if "Community Engagement" in content:
            return "Community Engagement"
        elif "Public Service" in content:
            return "Public Service"
        else:
            return "Public Service"  # Safe fallback
            
    except Exception as e:
        print(f"Classification error: {e}")
        return "Public Service"  # Safe fallback
```

#### **1.2 Extraction Function**
```python
async def extract_full_data(input_text, choice_text, model_text, system_text, user_text, collaboratory_text):
    """Full data extraction call - processes all fields with RAG"""
    try:
        # Combine retrieved content for full extraction
        full_context = f"{input_text} \n {choice_text} \n {model_text} \n {user_text} \n {collaboratory_text}"
        
        response = await llm_extraction.ainvoke([
            {"role": "system", "content": system_text},
            {"role": "user", "content": full_context}
        ])
        
        return response.content
        
    except Exception as e:
        print(f"Extraction error: {e}")
        return None
```

#### **1.3 Response Merging Function**
```python
def merge_responses(activity_type, extraction_response):
    """Merge classification and extraction responses"""
    try:
        # Extract JSON from extraction response
        extracted_json = extract_json_from_string(extraction_response)
        
        if "error" in extracted_json:
            return extracted_json
        
        # Ensure activityType is set correctly
        extracted_json["activityType"] = activity_type
        
        return extracted_json
        
    except Exception as e:
        print(f"Merge error: {e}")
        return {"error": "Failed to merge responses", "details": str(e)}
```

### **Phase 2: Main Function Modification**

#### **2.1 Current generate_activity Function**
```python
@app.post("/generate_activity")
@weave.op()
def generate_activity(input_data: InputData):  # Currently synchronous
    # ... existing code ...
    
    # Single LLM call (current approach)
    response = llm.invoke([
        {"role": "system", "content": system_text},
        {"role": "user", "content": full_context}
    ])
    
    # ... rest of processing ...
```

#### **2.2 Modified generate_activity Function**
```python
@app.post("/generate_activity")
@weave.op()
async def generate_activity(input_data: InputData):  # Now asynchronous
    import time
    start_time = time.time()
    
    # ... existing code for text extraction and RAG ...
    
    try:
        # Execute both tasks concurrently using asyncio
        classification_task = classify_activity_type(input_text)
        extraction_task = extract_full_data(input_text, choice_text, model_text, system_text, user_text, collaboratory_text)
        
        # Wait for both tasks to complete concurrently
        activity_type, extraction_response = await asyncio.gather(classification_task, extraction_task)
        
        # Combine classification and extraction responses
        final_response = merge_responses(activity_type, extraction_response)
        
        print(f"Parallel processing completed successfully")
        
    except Exception as e:
        print(f"Parallel processing failed, falling back to sequential: {e}")
        
        # Fallback to sequential processing
        activity_type = await classify_activity_type(input_text)
        extraction_response = await extract_full_data(input_text, choice_text, model_text, system_text, user_text, collaboratory_text)
        final_response = merge_responses(activity_type, extraction_response)
    
    # ... rest of processing with final_response ...
```

## 🔧 **Required Code Changes**

### **Change 1: Import asyncio**
```python
# Add to imports
import asyncio
```

### **Change 2: Initialize Dual Models**
```python
# Replace single LLM initialization
# llm = ChatOpenAI(model="gpt-4.1", temperature=0, openai_api_key=openai_api_key)

# With dual model setup
llm_classification = ChatOpenAI(model="gpt-4o-mini", temperature=0, openai_api_key=openai_api_key)
llm_extraction = ChatOpenAI(model="gpt-4.1", temperature=0, openai_api_key=openai_api_key)
```

### **Change 3: Add Classification Prompt Loading**
```python
def load_classification_prompt():
    try:
        with open("data/activity_type_classification_prompt.txt", "r", encoding="utf-8") as f:
            return f.read().strip()
    except FileNotFoundError:
        # Fallback prompt if file not found
        return """Classify as "Community Engagement" or "Public Service" based on evidence of shared planning/collaboration."""
```

### **Change 4: Modify Main Function Signature**
```python
# Change from:
def generate_activity(input_data: InputData):

# To:
async def generate_activity(input_data: InputData):
```

### **Change 5: Replace Single LLM Call with Parallel Processing**
```python
# Replace this:
response = llm.invoke([
    {"role": "system", "content": system_text},
    {"role": "user", "content": full_context}
])

# With this:
try:
    # Execute both tasks concurrently
    classification_task = classify_activity_type(input_text)
    extraction_task = extract_full_data(input_text, choice_text, model_text, system_text, user_text, collaboratory_text)
    
    # Wait for both tasks to complete concurrently
    activity_type, extraction_response = await asyncio.gather(classification_task, extraction_task)
    
    # Combine responses
    final_response = merge_responses(activity_type, extraction_response)
    
except Exception as e:
    # Fallback to sequential processing
    activity_type = await classify_activity_type(input_text)
    extraction_response = await extract_full_data(input_text, choice_text, model_text, system_text, user_text, collaboratory_text)
    final_response = merge_responses(activity_type, extraction_response)
```

### **Change 6: Update Response Processing**
```python
# Replace response.content with final_response
# Before:
ai_message = response.content

# After:
ai_message = final_response
```

## 📁 **File Structure Changes**

### **New Files to Create**
```
backend/
├── data/
│   └── activity_type_classification_prompt.txt  # NEW: Classification prompt
├── server.py                                    # MODIFIED: Add parallel processing
└── test_parallel.py                            # NEW: Test script
```

### **Files to Modify**
- `backend/server.py` - Add parallel processing functions and modify main function
- `backend/data/activity_type_classification_prompt.txt` - Create classification prompt

## 🎯 **Classification Prompt Content**

### **File: `backend/data/activity_type_classification_prompt.txt`**
```
You are a specialized classifier for determining activity types in community engagement activities.

Your task is to classify the given activity as either "Community Engagement" or "Public Service" based on these comprehensive rules:

🔍 CLASSIFICATION RULES:

**Community Engagement:**
Higher ed works with external, community-based partners (e.g., nonprofits, schools, agencies, neighborhoods) in a sustained, cooperative, or transformational partnership to co-create, co-plan, co-implement, or co-deliver shared goals.

**Public Service:**
Higher ed provides expertise, service, or assistance to external audiences or groups, with no evidence of shared planning, co-design, joint decision-making, or shared delivery.

📋 DECISION CRITERIA:

**Look for evidence of:**
1️⃣ **Shared planning, co-design, co-implementation, co-delivery, or sustained collaborative structure** between Higher ed and external community-based partners (beyond passive audiences or end users)

2️⃣ **Joint problem solving, shared learning, collaborative knowledge production, or collective action** where both Higher ed and external community partners benefit — not just by receiving a service

**Keywords to identify:**
- "partnership", "collaboration", "co-", "shared", "joint", "together"
- "co-create", "co-plan", "co-implement", "co-deliver"
- "sustained", "long-term", "transformational"
- "mutual learning", "collective action", "shared responsibility"

📌 DECISION RULE:
- If **BOTH criteria are met** → "Community Engagement"
- If **EITHER criterion is NOT met** → "Public Service"
- If **unclear** → Default to "Public Service"

🎯 OUTPUT:
Return ONLY the classification in this exact format:
```json
{
  "activityType": "Community Engagement"
}
```

OR

```json
{
  "activityType": "Public Service"
}
```

Do not include any other fields or explanations. Just the activityType classification.
```

## ⚡ **Performance Optimization Details**

### **Context Optimization Strategy**
```python
# Classification: Minimal context for speed
classification_context = input_text[:2000]  # First 2000 characters

# Extraction: Full context for accuracy
extraction_context = f"{input_text} \n {choice_text} \n {model_text} \n {user_text} \n {collaboratory_text}"
```

### **Model Selection Rationale**
- **Classification**: `gpt-4o-mini`
  - Faster response time (15-20 seconds)
  - Lower cost per token
  - Sufficient for simple classification task
  - Temperature: 0 (deterministic)

- **Extraction**: `gpt-4.1`
  - Higher reasoning capability
  - Better for complex data extraction
  - Maintains quality standards
  - Temperature: 0 (deterministic)

### **Expected Performance Improvements**
- **Response Time**: 1.5 min → 45-60 seconds
- **Improvement**: 50-67% faster
- **Cost**: 10-20% increase (two API calls)
- **Quality**: Maintained (same extraction logic)

## 🛡️ **Error Handling & Fallback**

### **Graceful Degradation Strategy**
```python
try:
    # Attempt parallel processing
    activity_type, extraction_response = await asyncio.gather(classification_task, extraction_task)
    print("Parallel processing completed successfully")
    
except Exception as e:
    print(f"Parallel processing failed, falling back to sequential: {e}")
    
    # Fallback to sequential processing
    activity_type = await classify_activity_type(input_text)
    extraction_response = await extract_full_data(input_text, choice_text, model_text, system_text, user_text, collaboratory_text)
    final_response = merge_responses(activity_type, extraction_response)
```

### **Fallback Benefits**
- **Reliability**: System never fails completely
- **Debugging**: Clear error logging
- **Performance**: Graceful degradation
- **User Experience**: Consistent response delivery

## 📊 **Monitoring & Metrics**

### **Performance Tracking**
```python
import time
start_time = time.time()

# ... parallel processing ...

total_time = time.time() - start_time
print(f"Total processing time: {total_time:.2f} seconds")

# Log to Weights & Biases
wandb.log({
    "ai_message": final_response,
    "structured_response": extracted_json,
    "total_processing_time": total_time,
    "parallel_processing": True
})
```

### **Metrics to Track**
- **Total Processing Time**: Start to finish duration
- **Parallel vs Sequential**: Success rates for each approach
- **Classification Time**: Time for activity type determination
- **Extraction Time**: Time for full data extraction
- **Token Usage**: Input/output ratios for each call

## 🧪 **Testing Strategy**

### **Unit Testing**
```python
# Test classification function
async def test_classification():
    result = await classify_activity_type("Sample text about partnership")
    assert result in ["Community Engagement", "Public Service"]

# Test extraction function
async def test_extraction():
    result = await extract_full_data("Sample text", "choice", "model", "system", "user", "collaboratory")
    assert result is not None

# Test response merging
def test_merging():
    result = merge_responses("Community Engagement", "Sample extraction response")
    assert "activityType" in result
```

### **Integration Testing**
```python
# Test complete parallel flow
async def test_parallel_flow():
    # Mock input data
    test_input = "Sample activity description"
    
    # Execute parallel processing
    classification_task = classify_activity_type(test_input)
    extraction_task = extract_full_data(test_input, ...)
    
    # Wait for both to complete
    activity_type, extraction_response = await asyncio.gather(classification_task, extraction_task)
    
    # Merge results
    final_response = merge_responses(activity_type, extraction_response)
    
    # Validate response
    assert final_response is not None
    assert "activityType" in final_response
```

## 🚀 **Deployment Steps**

### **Step 1: Create Classification Prompt**
```bash
# Create the classification prompt file
touch backend/data/activity_type_classification_prompt.txt
# Add content as specified above
```

### **Step 2: Modify Server.py**
```bash
# Add imports
# Add dual model initialization
# Add parallel processing functions
# Modify main function to be async
# Replace single LLM call with parallel processing
```

### **Step 3: Test Implementation**
```bash
cd backend
python test_parallel.py
```

### **Step 4: Deploy and Monitor**
```bash
# Start server
python server.py

# Monitor logs for:
# - "Parallel processing completed successfully"
# - "Total processing time: X.XX seconds"
# - Any fallback to sequential processing
```

## 🔍 **Troubleshooting Guide**

### **Common Issues & Solutions**

#### **Issue 1: Async/Await Errors**
```python
# Problem: Function not awaited properly
# Solution: Ensure all async functions are awaited
result = await classify_activity_type(input_text)  # Correct
result = classify_activity_type(input_text)        # Wrong
```

#### **Issue 2: Model Initialization Errors**
```python
# Problem: Models not initialized properly
# Solution: Check API keys and model names
llm_classification = ChatOpenAI(model="gpt-4o-mini", ...)
llm_extraction = ChatOpenAI(model="gpt-4.1", ...)
```

#### **Issue 3: Context Length Issues**
```python
# Problem: Classification context too long
# Solution: Limit context for classification
classification_context = input_text[:2000]  # Limit to 2000 chars
```

#### **Issue 4: Response Merging Failures**
```python
# Problem: Failed to merge responses
# Solution: Add error handling in merge function
try:
    # Merge logic
except Exception as e:
    return {"error": "Merge failed", "details": str(e)}
```

## 📈 **Expected Results**

### **Performance Metrics**
- **Response Time**: 1.5 min → 45-60 seconds
- **Throughput**: 2x improvement in requests per minute
- **User Experience**: Significantly faster response delivery
- **System Reliability**: Maintained with fallback mechanisms

### **Quality Metrics**
- **Classification Accuracy**: Same or better than current system
- **Data Extraction Quality**: Maintained (same logic)
- **Response Consistency**: Improved with structured merging
- **Error Rate**: Reduced with graceful fallback

## 🔮 **Future Enhancements**

### **Advanced Parallel Processing**
- **Multi-Model**: More than 2 parallel calls
- **Dynamic Splitting**: Adaptive task distribution
- **Load Balancing**: Intelligent request routing

### **Context Optimization**
- **Smart Truncation**: Intelligent context selection
- **Dynamic RAG**: Adaptive retrieval strategies
- **Context Caching**: Reuse similar contexts

### **Model Optimization**
- **Model Selection**: Dynamic model choice based on input complexity
- **Fine-tuning**: Custom model optimization for specific tasks
- **Ensemble Methods**: Multiple model combination for better accuracy

## 📋 **Implementation Checklist**

### **Pre-Implementation**
- [ ] Review current server.py architecture
- [ ] Identify all LLM calls to modify
- [ ] Prepare classification prompt content
- [ ] Set up testing environment
- [ ] Plan rollback strategy

### **Implementation**
- [ ] Add asyncio import
- [ ] Create dual model initialization
- [ ] Implement classification function
- [ ] Implement extraction function
- [ ] Implement response merging function
- [ ] Modify main function to be async
- [ ] Replace single LLM call with parallel processing
- [ ] Add error handling and fallback
- [ ] Add performance monitoring

### **Post-Implementation**
- [ ] Test with sample data
- [ ] Validate performance improvements
- [ ] Monitor error rates and fallback usage
- [ ] Collect user feedback
- [ ] Document lessons learned

## 💡 **Key Success Factors**

### **Technical Implementation**
- **Proper Async/Await**: Ensure all async functions are properly awaited
- **Error Handling**: Comprehensive error handling with graceful fallback
- **Context Optimization**: Right amount of context for each task
- **Response Validation**: Ensure merged responses are valid

### **Performance Optimization**
- **Model Selection**: Right tool for each job
- **Context Splitting**: Optimize context for each task
- **Parallel Execution**: True concurrent execution using asyncio
- **Monitoring**: Track performance improvements and issues

### **Quality Assurance**
- **Testing**: Comprehensive testing of all functions
- **Validation**: Ensure response quality is maintained
- **Fallback**: Reliable fallback to sequential processing
- **Monitoring**: Continuous performance and quality monitoring

---

*This implementation guide provides a complete roadmap for transforming your current sequential processing into efficient parallel processing while maintaining all existing functionality and quality standards.* 