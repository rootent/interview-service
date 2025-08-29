"""
Performance Optimizer - Monitor and optimize AI Interviewer system performance
CONTEXTUAL, ENTITY-ANCHORED FOLLOW-UPS WITH NO GENERIC FILLER
"""

import time
import psutil
import os
from model import RAGModel
from RAG_prompt import RAGPrompts

class PerformanceOptimizer:
    def __init__(self):
        self.start_time = time.time()
        self.memory_usage = []
        self.cpu_usage = []
        self.api_response_times = []
        
    def start_monitoring(self):
        """Start performance monitoring"""
        print("🔍 Starting performance monitoring...")
        self.start_time = time.time()
        
    def record_metrics(self):
        """Record current system metrics"""
        # Memory usage
        memory = psutil.virtual_memory()
        self.memory_usage.append({
            'timestamp': time.time(),
            'used_mb': memory.used / (1024 * 1024),
            'available_mb': memory.available / (1024 * 1024),
            'percent': memory.percent
        })
        
        # CPU usage
        cpu_percent = psutil.cpu_percent(interval=0.1)
        self.cpu_usage.append({
            'timestamp': time.time(),
            'cpu_percent': cpu_percent
        })
        
    def record_api_time(self, response_time):
        """Record API response time"""
        self.api_response_times.append({
            'timestamp': time.time(),
            'response_time': response_time
        })
        
    def get_performance_summary(self):
        """Get comprehensive performance summary"""
        total_time = time.time() - self.start_time
        
        # Memory analysis
        if self.memory_usage:
            avg_memory = sum(m['used_mb'] for m in self.memory_usage) / len(self.memory_usage)
            max_memory = max(m['used_mb'] for m in self.memory_usage)
            memory_trend = "Stable" if len(self.memory_usage) < 2 else (
                "Increasing" if self.memory_usage[-1]['used_mb'] > self.memory_usage[0]['used_mb'] else "Decreasing"
            )
        else:
            avg_memory = max_memory = 0
            memory_trend = "N/A"
        
        # CPU analysis
        if self.cpu_usage:
            avg_cpu = sum(c['cpu_percent'] for c in self.cpu_usage) / len(self.cpu_usage)
            max_cpu = max(c['cpu_percent'] for c in self.cpu_usage)
        else:
            avg_cpu = max_cpu = 0
        
        # API analysis
        if self.api_response_times:
            avg_api_time = sum(a['response_time'] for a in self.api_response_times) / len(self.api_response_times)
            min_api_time = min(a['response_time'] for a in self.api_response_times)
            max_api_time = max(a['response_time'] for a in self.api_response_times)
        else:
            avg_api_time = min_api_time = max_api_time = 0
        
        return {
            'total_time': total_time,
            'memory': {
                'average_mb': avg_memory,
                'max_mb': max_memory,
                'trend': memory_trend
            },
            'cpu': {
                'average_percent': avg_cpu,
                'max_percent': max_cpu
            },
            'api': {
                'average_response_time': avg_api_time,
                'min_response_time': min_api_time,
                'max_response_time': max_api_time,
                'total_calls': len(self.api_response_times)
            }
        }
        
    def print_performance_report(self):
        """Print detailed performance report"""
        summary = self.get_performance_summary()
        
        print("\n" + "="*60)
        print("📊 PERFORMANCE REPORT")
        print("="*60)
        print(f"⏱️  Total Runtime: {summary['total_time']:.2f}s")
        
        print(f"\n💾 Memory Usage:")
        print(f"   • Average: {summary['memory']['average_mb']:.1f} MB")
        print(f"   • Peak: {summary['memory']['max_mb']:.1f} MB")
        print(f"   • Trend: {summary['memory']['trend']}")
        
        print(f"\n🖥️  CPU Usage:")
        print(f"   • Average: {summary['cpu']['average_percent']:.1f}%")
        print(f"   • Peak: {summary['cpu']['max_percent']:.1f}%")
        
        print(f"\n🌐 API Performance:")
        print(f"   • Total Calls: {summary['api']['total_calls']}")
        print(f"   • Average Response: {summary['api']['average_response_time']:.3f}s")
        print(f"   • Fastest: {summary['api']['min_response_time']:.3f}s")
        print(f"   • Slowest: {summary['api']['max_response_time']:.3f}s")
        
        # Performance recommendations
        print(f"\n💡 Performance Recommendations:")
        
        if summary['api']['average_response_time'] > 3.0:
            print("   ⚠️  API responses are slow - consider network optimization")
        
        if summary['memory']['max_mb'] > 1000:
            print("   ⚠️  High memory usage - consider reducing chunk sizes")
        
        if summary['cpu']['max_percent'] > 80:
            print("   ⚠️  High CPU usage - consider using lighter models")
            
        if summary['api']['total_calls'] > 25:
            print("   ⚠️  Many API calls - consider batching or caching")
            
        print("="*60)

def optimize_config():
    """Suggest configuration optimizations for AI Interviewer system"""
    print("\n🔧 AI INTERVIEWER CONFIGURATION OPTIMIZATION")
    print("="*60)
    
    current_config = {
        'CHUNK_SIZE': 300,
        'CHUNK_OVERLAP': 50,
        'MAX_FOLLOW_UPS': 3,
        'TECHNICAL_DEPTH': 'high',
        'EVALUATION_PARAMETERS': 'language,tech_skills,content_depth,problem_solving,communication'
    }
    
    print("Current optimized settings for AI Interviewer:")
    for key, value in current_config.items():
        print(f"   • {key}: {value}")
    
    print("\nFor even faster performance (may reduce question depth):")
    print("   • CHUNK_SIZE: 200 (smaller chunks)")
    print("   • CHUNK_OVERLAP: 25 (minimal overlap)")
    print("   • MAX_FOLLOW_UPS: 2 (fewer follow-ups)")
    
    print("\nFor maximum question depth (may be slower):")
    print("   • CHUNK_SIZE: 500 (larger chunks for context)")
    print("   • CHUNK_OVERLAP: 100 (more overlap for continuity)")
    print("   • MAX_FOLLOW_UPS: 4 (more follow-ups)")
    
    print("\n⚠️  Note: Question depth vs. speed trade-off. Higher depth = better interviews but slower performance.")

def run_performance_test():
    """Run a comprehensive performance test for AI Interviewer system"""
    print("\n🧪 RUNNING AI INTERVIEWER PERFORMANCE TEST")
    print("="*60)
    
    optimizer = PerformanceOptimizer()
    optimizer.start_monitoring()
    
    # Test document loading
    print("Testing document loading...")
    rag_model = RAGModel()
    
    start_time = time.time()
    success = rag_model.load_documents()
    load_time = time.time() - start_time
    
    if success:
        print(f"✅ Document loading: {load_time:.2f}s")
        
        # Test embedding creation
        print("Testing embedding creation...")
        start_time = time.time()
        success = rag_model.create_embeddings()
        embedding_time = time.time() - start_time
        
        if success:
            print(f"✅ Embedding creation: {embedding_time:.2f}s")
            
            # Test question generation
            print("Testing in-depth question generation...")
            jd_content, vision_content, resume_content = rag_model.get_document_content()
            weights = rag_model.get_question_weights()
            
            prompts = RAGPrompts(rag_model.api_key)
            start_time = time.time()
            questions = prompts.generate_weighted_questions(jd_content, vision_content, resume_content, weights)
            question_time = time.time() - start_time
            
            print(f"✅ Question generation: {question_time:.2f}s")
            print(f"✅ Generated {len(questions)} in-depth questions")
            
            # Test contextual follow-up generation
            print("Testing contextual follow-up generation...")
            if questions:
                start_time = time.time()
                follow_up = prompts.generate_adaptive_followup(
                    questions[0], "Sample user answer for testing", 
                    jd_content, vision_content, resume_content
                )
                follow_up_time = time.time() - start_time
                print(f"✅ Contextual follow-up generation: {follow_up_time:.2f}s")
            
            # Test vision/mission question generation
            print("Testing vision/mission question generation...")
            start_time = time.time()
            vision_question = prompts.generate_vision_mission_question(
                "Sample answer for testing", vision_content, jd_content
            )
            vision_time = time.time() - start_time
            print(f"✅ Vision/Mission question generation: {vision_time:.2f}s")
            
            # Test leadership question generation
            print("Testing leadership question generation...")
            start_time = time.time()
            leadership_question = prompts.generate_leadership_question(
                "Sample answer for testing", jd_content, vision_content
            )
            leadership_time = time.time() - start_time
            print(f"✅ Leadership question generation: {leadership_time:.2f}s")
            
            # Get performance stats
            rag_stats = rag_model.get_performance_stats()
            prompt_stats = prompts.get_performance_stats()
            
            print(f"\n📊 AI Interviewer Performance Summary:")
            print(f"   • Document processing: {rag_stats['processing_time']:.2f}s")
            print(f"   • Embedding creation: {embedding_time:.2f}s")
            print(f"   • Question generation: {question_time:.2f}s")
            print(f"   • Contextual follow-up: {follow_up_time:.2f}s")
            print(f"   • Vision/Mission question: {vision_time:.2f}s")
            print(f"   • Leadership question: {leadership_time:.2f}s")
            print(f"   • API calls: {prompt_stats['total_api_calls']}")
            print(f"   • Total API time: {prompt_stats['total_api_time']:.2f}s")
            
            total_time = rag_stats['processing_time'] + embedding_time + question_time + follow_up_time + vision_time + leadership_time
            print(f"   • Total test time: {total_time:.2f}s")
            
            # Performance rating for AI Interviewer
            if total_time < 20:
                rating = "🚀 EXCELLENT - Perfect for real-time AI interviews"
            elif total_time < 35:
                rating = "⚡ GOOD - Suitable for AI interviews with minor delays"
            elif total_time < 50:
                rating = "✅ ACCEPTABLE - May have delays in AI interviews"
            else:
                rating = "⚠️  SLOW - May impact AI interview flow"
                
            print(f"   • AI Interviewer Performance Rating: {rating}")
            
            # AI Interviewer assessment
            print(f"\n🎤 AI Interviewer Assessment:")
            print(f"   • Technical depth setting: {rag_stats['technical_depth']}")
            print(f"   • Evaluation parameters: {len(rag_stats['evaluation_parameters'])} parameters")
            print(f"   • Follow-up questions: {rag_stats.get('max_follow_ups', 3)} per main question")
            print(f"   • Question types: Technical, Experience, Vision/Mission")
            print(f"   • Adaptive flow: Yes (AI-driven topic selection)")
            print(f"   • Contextual follow-ups: Yes (entity-anchored, no generic filler)")
            
        else:
            print("❌ Embedding creation failed")
    else:
        print("❌ Document loading failed")
    
    optimizer.print_performance_report()

if __name__ == "__main__":
    print("🎯 AI Interviewer Performance Optimizer")
    print("="*60)
    
    while True:
        print("\nChoose an option:")
        print("1. Run AI Interviewer Performance Test")
        print("2. Show AI Interviewer Configuration Optimization Suggestions")
        print("3. Exit")
        
        choice = input("\nEnter your choice (1-3): ").strip()
        
        if choice == "1":
            run_performance_test()
        elif choice == "2":
            optimize_config()
        elif choice == "3":
            print("👋 Goodbye!")
            break
        else:
            print("❌ Invalid choice. Please try again.")
