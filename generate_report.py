import json
import statistics
from datetime import datetime
from typing import Dict
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT


class PerformanceReportGenerator:
    """Generate PDF performance report"""
    
    def __init__(self, benchmark_data_file='benchmark_results.json'):
        self.benchmark_data = self._load_benchmark_data(benchmark_data_file)
        self.styles = getSampleStyleSheet()
        self._create_custom_styles()
    
    def _load_benchmark_data(self, filename: str) -> Dict:
        """Load benchmark data from JSON"""
        try:
            with open(filename, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"Error: {filename} not found. Run benchmark_load.py first.")
            raise
    
    def _create_custom_styles(self):
        """Create custom paragraph styles"""
        self.styles.add(ParagraphStyle(
            name='CustomTitle',
            parent=self.styles['Heading1'],
            fontSize=24,
            textColor=colors.HexColor('#1f4788'),
            spaceAfter=30,
            alignment=TA_CENTER,
            fontName='Helvetica-Bold'
        ))
        
        self.styles.add(ParagraphStyle(
            name='CustomHeading',
            parent=self.styles['Heading2'],
            fontSize=14,
            textColor=colors.HexColor('#2E5C8A'),
            spaceAfter=12,
            spaceBefore=12,
            fontName='Helvetica-Bold'
        ))
        
        self.styles.add(ParagraphStyle(
            name='CustomBody',
            parent=self.styles['BodyText'],
            fontSize=10,
            leading=14
        ))
    
    def _create_title_page(self):
        """Create report title and metadata"""
        elements = []
        
        title = Paragraph(
            "Matching Engine Performance Report",
            self.styles['CustomTitle']
        )
        elements.append(title)
        elements.append(Spacer(1, 0.2*inch))
        
        timestamp = datetime.fromisoformat(self.benchmark_data['timestamp'])
        subtitle = Paragraph(
            f"Generated: {timestamp.strftime('%Y-%m-%d %H:%M:%S')}<br/>v3.0 - High Performance Benchmark",
            self.styles['Normal']
        )
        elements.append(subtitle)
        elements.append(Spacer(1, 0.5*inch))
        
        # Summary metrics
        summary_data = [
            ['Metric', 'Value'],
            ['Total Requests', str(self.benchmark_data['total_requests'])],
            ['Total Errors', str(self.benchmark_data['total_errors'])],
            ['Success Rate', f"{(1 - self.benchmark_data['total_errors']/max(1, self.benchmark_data['total_requests']))*100:.1f}%"],
        ]
        
        summary_table = Table(summary_data, colWidths=[3*inch, 2*inch])
        summary_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2E5C8A')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 11),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#F0F0F0')]),
        ]))
        
        elements.append(summary_table)
        elements.append(Spacer(1, 0.3*inch))
        
        return elements
    
    def _create_latency_tables(self):
        """Create latency statistics tables"""
        elements = []
        
        elements.append(Paragraph("Performance Metrics", self.styles['CustomHeading']))
        elements.append(Spacer(1, 0.15*inch))
        
        results = self.benchmark_data['results']
        
        for operation, stats in results.items():
            # Operation section
            elements.append(Paragraph(f"{operation.replace('_', ' ').title()}", self.styles['Heading3']))
            
            # Stats table
            table_data = [
                ['Metric', 'Value (ms)', 'Metric', 'Value (ms)'],
                ['Min', f"{stats['min']:.2f}", 'P50', f"{stats['p50']:.2f}"],
                ['Mean', f"{stats['mean']:.2f}", 'P95', f"{stats['p95']:.2f}"],
                ['Median', f"{stats['median']:.2f}", 'P99', f"{stats['p99']:.2f}"],
                ['Max', f"{stats['max']:.2f}", 'StdDev', f"{stats['stdev']:.2f}"],
                ['Count', str(stats['count']), '', ''],
            ]
            
            table = Table(table_data, colWidths=[1.5*inch, 1.3*inch, 1.5*inch, 1.3*inch])
            table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#4472C4')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 10),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 10),
                ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#E7E6E6')),
                ('GRID', (0, 0), (-1, -1), 1, colors.grey),
                ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#F5F5F5')]),
                ('FONTSIZE', (0, 1), (-1, -1), 9),
            ]))
            
            elements.append(table)
            elements.append(Spacer(1, 0.15*inch))
        
        return elements
    
    def _create_summary_section(self):
        """Create performance summary and interpretation"""
        elements = []
        
        elements.append(PageBreak())
        elements.append(Paragraph("Performance Analysis", self.styles['CustomHeading']))
        elements.append(Spacer(1, 0.15*inch))
        
        results = self.benchmark_data['results']
        
        # Find best and worst performing operations
        if results:
            best_op = min(results.items(), key=lambda x: x[1]['p50'])
            worst_op = max(results.items(), key=lambda x: x[1]['p50'])
            
            summary_text = f"""
            <b>Key Findings:</b><br/>
            <br/>
            • <b>Fastest Operation (P50):</b> {best_op[0].replace('_', ' ').title()} at {best_op[1]['p50']:.2f}ms<br/>
            • <b>Slowest Operation (P50):</b> {worst_op[0].replace('_', ' ').title()} at {worst_op[1]['p50']:.2f}ms<br/>
            • <b>Average P95 Latency:</b> {statistics.mean([s['p95'] for s in results.values()]):.2f}ms<br/>
            • <b>Average P99 Latency:</b> {statistics.mean([s['p99'] for s in results.values()]):.2f}ms<br/>
            <br/>
            <b>Interpretation:</b><br/>
            P50 (Median) represents typical performance under normal conditions.<br/>
            P95 and P99 represent worst-case performance for 95% and 99% of requests.<br/>
            Lower P95/P99 values indicate more consistent performance.<br/>
            <br/>
            <b>Recommendations:</b><br/>
            • Operations with P95 > 500ms should be optimized<br/>
            • Consider caching for frequently accessed endpoints<br/>
            • Monitor P99 values for SLA compliance<br/>
            """
            
            elements.append(Paragraph(summary_text, self.styles['CustomBody']))
        
        return elements
    
    def _create_system_info(self):
        """Create system information section"""
        elements = []
        
        elements.append(Spacer(1, 0.3*inch))
        elements.append(Paragraph("System Information", self.styles['CustomHeading']))
        elements.append(Spacer(1, 0.15*inch))
        
        info_text = """
        <b>Matching Engine:</b> v3.0<br/>
        <b>API Version:</b> REST API v3<br/>
        <b>WebSocket Support:</b> Enabled<br/>
        <b>Database:</b> In-memory with persistence<br/>
        <b>Concurrency:</b> Async/await pattern<br/>
        <b>Framework:</b> FastAPI + Uvicorn<br/>
        <br/>
        <b>Test Environment:</b><br/>
        • Load Type: Sequential + Concurrent<br/>
        • Order Types: Market, Limit, IOC, FOK, Stop Orders<br/>
        • Symbols Tested: BTC-USDT, ETH-USDT<br/>
        """
        
        elements.append(Paragraph(info_text, self.styles['CustomBody']))
        
        return elements
    
    def generate_pdf(self, output_file='performance_report.pdf'):
        """Generate complete PDF report"""
        print(f"Generating PDF report: {output_file}")
        
        doc = SimpleDocTemplate(
            output_file,
            pagesize=letter,
            rightMargin=0.5*inch,
            leftMargin=0.5*inch,
            topMargin=0.75*inch,
            bottomMargin=0.75*inch
        )
        
        # Build document
        story = []
        
        # Title page
        story.extend(self._create_title_page())
        
        # Latency tables
        story.extend(self._create_latency_tables())
        
        # Analysis
        story.extend(self._create_summary_section())
        
        # System info
        story.extend(self._create_system_info())
        
        # Build PDF
        doc.build(story)
        
        print(f"✓ PDF report generated: {output_file}")
        print(f"  File size: {len(open(output_file, 'rb').read()) / 1024:.1f} KB")


if __name__ == "__main__":
    try:
        generator = PerformanceReportGenerator('benchmark_results.json')
        generator.generate_pdf('performance_report.pdf')
        print("\nReport generation completed successfully!")
    except Exception as e:
        print(f"Error generating report: {e}")