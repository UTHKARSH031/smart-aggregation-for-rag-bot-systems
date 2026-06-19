"""
Sample Data Generator
=====================

Creates sample financial documents and questions for testing.
Mimics FinanceBench structure. Extended to 5 docs, 15 questions.
"""

SAMPLE_DOCUMENTS = {
    "doc1": """TECHCORP INC. - ANNUAL REPORT 2023

EXECUTIVE SUMMARY

TechCorp Inc. achieved record revenue in fiscal year 2023, driven by strong performance across all business segments. 
Total revenue reached $2.5 billion, representing a 22% increase year-over-year from $2.05 billion in 2022.

FINANCIAL HIGHLIGHTS

Q4 2023 PERFORMANCE

Revenue for the fourth quarter of 2023 was $750 million, up 18% from Q4 2022 revenue of $636 million. 
The increase was primarily driven by cloud services revenue, which grew 45% to $340 million.

Operating income for Q4 2023 was $180 million, compared to $145 million in the prior year quarter, 
representing a 24% increase. Net income for the quarter was $135 million, or $1.25 per diluted share.

FULL YEAR 2023 RESULTS

| Metric | 2023 | 2022 | Change |
|--------|------|------|--------|
| Total Revenue | $2,500M | $2,050M | +22% |
| Cloud Services | $1,200M | $820M | +46% |
| Enterprise Software | $900M | $850M | +6% |
| Professional Services | $400M | $380M | +5% |
| Operating Income | $625M | $485M | +29% |
| Net Income | $475M | $365M | +30% |

SEGMENT PERFORMANCE

CLOUD SERVICES SEGMENT

Cloud services revenue increased 46% to $1,200 million in 2023, compared to $820 million in 2022. 
This growth was driven by:
- 35% increase in subscription revenue from existing customers
- 28% increase in new customer acquisitions
- Expansion into European and Asian markets

Annual recurring revenue (ARR) from cloud services reached $1,450 million as of December 31, 2023, 
up from $980 million at the end of 2022.

ENTERPRISE SOFTWARE SEGMENT

Enterprise software revenue grew 6% to $900 million, compared to $850 million in the prior year.
Key drivers included:
- Launch of TechCorp AI Suite in March 2023
- Strong renewal rates of 94%
- Average deal size increased 12% to $185,000

REGIONAL PERFORMANCE

| Region | Revenue 2023 | Revenue 2022 | Growth |
|--------|--------------|--------------|--------|
| North America | $1,500M | $1,280M | +17% |
| Europe | $650M | $520M | +25% |
| Asia-Pacific | $350M | $250M | +40% |

North America remains our largest market, representing 60% of total revenue. However, international 
markets are growing faster, with Asia-Pacific showing particularly strong momentum.

OPERATING EXPENSES

Total operating expenses for 2023 were $1,875 million, compared to $1,565 million in 2022, an increase of 20%.

Research and development expenses increased 28% to $625 million, reflecting our continued investment 
in AI and machine learning capabilities.

Sales and marketing expenses were $750 million, up 18% from $635 million in the prior year.

General and administrative expenses increased 12% to $500 million.

CASH FLOW AND BALANCE SHEET

Operating cash flow for 2023 was $580 million, compared to $445 million in 2022.
Free cash flow was $480 million, up from $360 million in the prior year.

As of December 31, 2023:
- Cash and cash equivalents: $850 million
- Total assets: $3,200 million
- Total liabilities: $1,100 million
- Stockholders' equity: $2,100 million

OUTLOOK FOR 2024

We expect continued strong growth in 2024, with projected total revenue of $2,950 million to $3,050 million,
representing 18-22% growth over 2023.

Cloud services are expected to grow 35-40%, while enterprise software should grow 8-12%.

We plan to invest an additional $150 million in R&D in 2024 to accelerate our AI initiatives.
""",

    "doc2": """GLOBALBANK CORPORATION - 10-K FILING 2023

BUSINESS OVERVIEW

GlobalBank Corporation is a diversified financial services company providing banking, investment, 
and wealth management services to individuals, corporations, and institutions worldwide.

FINANCIAL SUMMARY 2023

Total revenue for 2023 was $45.2 billion, an increase of 8% from $41.8 billion in 2022.
Net interest income was $28.5 billion, up 12% year-over-year.
Non-interest income was $16.7 billion, up 2% from the prior year.

Net income for 2023 was $12.8 billion, or $4.85 per diluted share, compared to $11.2 billion, 
or $4.20 per diluted share, in 2022.

QUARTERLY PERFORMANCE - Q4 2023

Fourth quarter revenue was $11.8 billion, up 9% from Q4 2022 revenue of $10.8 billion.

Q4 net income was $3.2 billion, compared to $2.9 billion in Q4 2022.

| Q4 2023 Segment | Revenue | YoY Change |
|-----------------|---------|------------|
| Consumer Banking | $4,200M | +6% |
| Corporate Banking | $3,800M | +12% |
| Investment Banking | $2,500M | +8% |
| Wealth Management | $1,300M | +10% |

BUSINESS SEGMENT ANALYSIS

CONSUMER BANKING

Consumer banking revenue for 2023 was $16.5 billion, an increase of 7% from $15.4 billion in 2022.
The segment served 28 million customers as of year-end, up from 26 million in 2022.

Average deposits per customer increased 9% to $18,500.
Loan originations totaled $42 billion in 2023, up 11% year-over-year.

CORPORATE BANKING

Corporate banking revenue increased 11% to $15.2 billion in 2023.
Commercial loan portfolio grew 13% to $285 billion.
Middle market lending increased 15%, while large corporate lending grew 10%.

INVESTMENT BANKING

Investment banking revenue was $9.3 billion, up 8% from $8.6 billion in 2022.
Advisory fees increased 15% to $3.8 billion.
Underwriting revenue grew 5% to $3.2 billion.
Trading revenue was $2.3 billion, relatively flat year-over-year.

WEALTH MANAGEMENT

Wealth management revenue increased 9% to $5.2 billion.
Assets under management (AUM) grew 12% to $1.2 trillion.
Client count increased 8% to 450,000 high-net-worth clients.

CREDIT QUALITY

Non-performing loans as a percentage of total loans decreased to 0.68% from 0.82% in 2022.
Net charge-offs were 0.42% of average loans, down from 0.51% in the prior year.

Allowance for credit losses was $8.2 billion, or 1.8% of total loans, compared to $8.5 billion, 
or 2.0%, at the end of 2022.

CAPITAL AND LIQUIDITY

Common Equity Tier 1 (CET1) ratio was 12.5% as of December 31, 2023, well above regulatory requirements.
Total capital ratio was 15.8%.

Liquidity coverage ratio (LCR) was 125%, exceeding the 100% regulatory minimum.

Return on equity (ROE) for 2023 was 14.2%, compared to 13.1% in 2022.
Return on assets (ROA) was 1.15%, up from 1.05% in the prior year.
""",

    "doc3": """RETAIL SOLUTIONS INC. - QUARTERLY REPORT Q3 2023

COMPANY OVERVIEW

Retail Solutions Inc. is a leading provider of point-of-sale systems, inventory management software,
and e-commerce platforms for retail businesses.

THIRD QUARTER 2023 RESULTS

Total revenue for Q3 2023 was $285 million, an increase of 16% compared to $246 million in Q3 2022.

Subscription revenue was $195 million, up 24% year-over-year, now representing 68% of total revenue.
Professional services revenue was $55 million, up 8%.
Hardware sales were $35 million, down 5% as we transition to a software-first model.

| Q3 2023 Metrics | Value |
|-----------------|-------|
| Total Revenue | $285M |
| Subscription ARR | $820M |
| Gross Margin | 72% |
| Operating Margin | 18% |
| Net Income | $38M |
| EPS (diluted) | $0.95 |

SUBSCRIPTION BUSINESS PERFORMANCE

Annual Recurring Revenue (ARR) reached $820 million as of September 30, 2023, up 26% from $650 million
a year ago.

Net revenue retention rate was 118%, indicating strong upsell and expansion within existing customer base.

Customer count grew to 45,000, up from 38,000 in Q3 2022, representing 18% growth.
Average revenue per customer increased 7% to $18,200 annually.

PRODUCT DEVELOPMENTS

Launched RetailOS 3.0 in July 2023, featuring:
- AI-powered inventory predictions
- Real-time analytics dashboard
- Enhanced mobile POS capabilities
- Integration with major e-commerce platforms

Early adoption has been strong, with 2,800 customers upgrading in the first 90 days.

CUSTOMER METRICS

| Customer Segment | Count | ARR Contribution |
|------------------|-------|------------------|
| Small Business | 32,000 | $256M |
| Mid-Market | 11,000 | $385M |
| Enterprise | 2,000 | $179M |

Enterprise customers, while only 4% of total count, contribute 22% of ARR, with average contract
value of $89,500.

GEOGRAPHIC EXPANSION

International revenue grew 35% to $68 million, now representing 24% of total revenue.
Opened offices in London and Sydney in Q3 2023.
Signed partnership with European distributor covering 15 countries.

EXPENSES AND PROFITABILITY

Gross margin improved to 72% from 68% in Q3 2022, driven by higher-margin subscription mix.

Operating expenses were $154 million, up 14% year-over-year:
- R&D: $52M (+22%)
- Sales & Marketing: $68M (+12%)  
- G&A: $34M (+8%)

Operating income was $51 million, up 22% from $42 million in Q3 2022.
Operating margin expanded to 18% from 17% in the prior year quarter.

CASH FLOW

Operating cash flow for Q3 was $58 million, compared to $45 million in Q3 2022.
Free cash flow was $48 million, up from $36 million in the prior year quarter.

Ended the quarter with $180 million in cash and no debt.
""",

    "doc4": """HEALTHTECH INNOVATIONS LLC - ANNUAL REPORT 2023

COMPANY BACKGROUND

HealthTech Innovations LLC is a medical technology company specializing in AI-driven diagnostic tools,
electronic health record (EHR) platforms, and remote patient monitoring systems.

FINANCIAL PERFORMANCE 2023

Total revenue for fiscal year 2023 was $1.1 billion, up 31% from $840 million in 2022.

REVENUE BREAKDOWN

| Product Line | 2023 Revenue | 2022 Revenue | Growth |
|---|---|---|---|
| AI Diagnostics | $410M | $280M | +46% |
| EHR Platform | $380M | $320M | +19% |
| Remote Monitoring | $210M | $160M | +31% |
| Services & Support | $100M | $80M | +25% |

GROSS MARGIN AND PROFITABILITY

Gross margin for 2023 was 68%, compared to 64% in 2022, driven by the increasing share of
high-margin software and subscription revenues.

Operating income was $176 million, representing a 16% operating margin.
Net income was $132 million, or $2.64 per diluted share, up from $88 million in 2022.

EBITDA reached $220 million, an EBITDA margin of 20%.

CUSTOMER AND MARKET METRICS

Total hospital system customers: 820, up from 640 in 2022.
Total clinic and physician group customers: 14,500, up 28% year-over-year.

AI Diagnostics platform processed 48 million imaging studies in 2023, a 55% increase from 2022.
EHR platform active users grew to 92,000 clinicians.

Average contract value for enterprise hospital systems: $1.2 million annually.
Net revenue retention rate: 122%.

REGULATORY AND CLINICAL UPDATES

Received FDA 510(k) clearance for our AI-powered radiology assistant in March 2023.
Achieved HITRUST certification for our cloud-hosted EHR platform.
Published clinical validation data showing 94% diagnostic accuracy vs 87% for traditional methods.

RESEARCH AND DEVELOPMENT

R&D expenditure was $198 million, representing 18% of revenue.
Filed 23 new patents in 2023 in AI diagnostics and predictive analytics.
Launched clinical trials for AI-based early disease detection across 12 hospital partners.

BALANCE SHEET HIGHLIGHTS

Cash and equivalents: $320 million.
Total assets: $1.8 billion.
Total debt: $250 million (all long-term, at fixed 4.2% interest rate).
Stockholders' equity: $1.1 billion.

OUTLOOK 2024

Revenue guidance: $1.38 billion to $1.42 billion (25-29% growth).
Operating margin target: 18-20%.
Planned R&D investment: $240 million.
Expected new hospital system customer additions: 150-180.
""",

    "doc5": """ENERGYSMART CORP - ESG & FINANCIAL REPORT 2023

EXECUTIVE OVERVIEW

EnergySmart Corp is a renewable energy and smart grid technology company. We develop and operate
solar farms, wind installations, and AI-driven energy management platforms for utilities and large
commercial customers.

2023 FINANCIAL SUMMARY

Total revenue: $3.8 billion, up 28% from $2.97 billion in 2022.

| Segment | 2023 Revenue | 2022 Revenue | Change |
|---------|-------------|-------------|--------|
| Solar Generation | $1,620M | $1,180M | +37% |
| Wind Generation | $980M | $820M | +20% |
| Smart Grid Platform | $740M | $560M | +32% |
| Energy Services | $460M | $410M | +12% |

OPERATIONAL METRICS

Total installed renewable capacity: 8.4 GW, up from 6.2 GW in 2022.
Electricity generated in 2023: 22.1 TWh, up 35% year-over-year.
New capacity added in 2023: 2.2 GW (1.4 GW solar, 0.8 GW wind).
Smart grid customers managed: 3.1 million endpoints.

EBITDA AND PROFITABILITY

Adjusted EBITDA: $1.14 billion, margin of 30%.
Operating income: $760 million.
Net income: $418 million, or $3.48 per diluted share.

CAPEX AND INVESTMENTS

Capital expenditure: $2.1 billion (primarily new project development).
Project pipeline: 12.5 GW of renewable projects in development or construction.
Battery storage deployments: 400 MWh in 2023, cumulative 950 MWh.

ESG PERFORMANCE

Carbon offset generated from our renewable assets: 18.4 million metric tons CO2e in 2023.
Scope 1 & 2 emissions from company operations: 42,000 metric tons CO2e (down 18% YoY).
Employees: 7,800 globally, up 22% from 2022.
Renewable energy in our own operations: 100% since 2021.
Water usage intensity: reduced 14% vs 2022 baseline.

DEBT AND FINANCING

Total debt: $4.2 billion (primarily long-term project finance at avg 5.1% rate).
Project finance facilities: $3.6 billion secured against operating renewable assets.
Corporate revolving credit: $600 million (undrawn as of year-end).

DIVIDEND AND CAPITAL RETURNS

Dividend per share: $1.20 (paid quarterly at $0.30/share).
Share buyback: $150 million of common shares repurchased in 2023.

2024 OUTLOOK

Revenue guidance: $4.6 billion to $4.8 billion (21-26% growth).
New renewable capacity target: 3.0 GW.
EBITDA margin target: 31-33%.
Planned CAPEX: $2.6 billion.
"""
}

SAMPLE_QUESTIONS = [
    # doc1 questions
    {
        'question': "What was TechCorp's Q4 2023 revenue?",
        'answer': '$750 million',
        'doc_id': 'doc1',
        'type': 'factual'
    },
    {
        'question': "Calculate TechCorp's year-over-year revenue growth rate from 2022 to 2023.",
        'answer': '22%',
        'doc_id': 'doc1',
        'type': 'computational'
    },
    {
        'question': "What was TechCorp's cloud services revenue in 2023?",
        'answer': '$1,200 million',
        'doc_id': 'doc1',
        'type': 'factual'
    },
    # doc2 questions
    {
        'question': "What was GlobalBank's total revenue in 2023?",
        'answer': '$45.2 billion',
        'doc_id': 'doc2',
        'type': 'factual'
    },
    {
        'question': "What was GlobalBank's Q4 2023 net income?",
        'answer': '$3.2 billion',
        'doc_id': 'doc2',
        'type': 'factual'
    },
    {
        'question': "What was GlobalBank's return on equity (ROE) in 2023?",
        'answer': '14.2%',
        'doc_id': 'doc2',
        'type': 'factual'
    },
    {
        'question': "What was GlobalBank's CET1 capital ratio at end of 2023?",
        'answer': '12.5%',
        'doc_id': 'doc2',
        'type': 'factual'
    },
    # doc3 questions
    {
        'question': "What was Retail Solutions' Q3 2023 total revenue?",
        'answer': '$285 million',
        'doc_id': 'doc3',
        'type': 'factual'
    },
    {
        'question': "What is Retail Solutions' Annual Recurring Revenue (ARR) as of Q3 2023?",
        'answer': '$820 million',
        'doc_id': 'doc3',
        'type': 'factual'
    },
    {
        'question': "Calculate Retail Solutions' subscription revenue growth rate in Q3 2023.",
        'answer': '24%',
        'doc_id': 'doc3',
        'type': 'computational'
    },
    # doc4 questions
    {
        'question': "What was HealthTech Innovations' total revenue in 2023?",
        'answer': '$1.1 billion',
        'doc_id': 'doc4',
        'type': 'factual'
    },
    {
        'question': "What was HealthTech's net revenue retention rate in 2023?",
        'answer': '122%',
        'doc_id': 'doc4',
        'type': 'factual'
    },
    {
        'question': "How much did HealthTech spend on R&D in 2023?",
        'answer': '$198 million (18% of revenue)',
        'doc_id': 'doc4',
        'type': 'factual'
    },
    # doc5 questions
    {
        'question': "What was EnergySmart Corp's total revenue in 2023?",
        'answer': '$3.8 billion',
        'doc_id': 'doc5',
        'type': 'factual'
    },
    {
        'question': "What was EnergySmart's total installed renewable capacity at end of 2023?",
        'answer': '8.4 GW',
        'doc_id': 'doc5',
        'type': 'factual'
    },
]


def get_sample_data(num_docs=30):
    """Returns sample documents and questions"""
    docs = dict(SAMPLE_DOCUMENTS)
    qs = list(SAMPLE_QUESTIONS)
    
    current_doc_count = len(docs)
    for i in range(current_doc_count + 1, num_docs + 1):
        doc_id = f"doc{i}"
        docs[doc_id] = f"COMPANY {i} - ANNUAL REPORT 2023\n\nEXECUTIVE OVERVIEW\n\nCompany {i} achieved record revenue in fiscal year 2023, driven by strong performance across all business segments.\nTotal revenue reached ${100 + i * 10} million, representing a 15% increase year-over-year.\n\nFINANCIAL HIGHLIGHTS\n\nNet income for the quarter was ${20 + i} million. The company expects continued strong growth in 2024."
        
        qs.append({
            'question': f"What was Company {i}'s total revenue in 2023?",
            'answer': f"${100 + i * 10} million",
            'doc_id': doc_id,
            'type': 'factual'
        })
        qs.append({
            'question': f"What was Company {i}'s net income for the quarter?",
            'answer': f"${20 + i} million",
            'doc_id': doc_id,
            'type': 'factual'
        })
        qs.append({
            'question': f"What is the revenue growth rate for Company {i}?",
            'answer': "15%",
            'doc_id': doc_id,
            'type': 'factual'
        })
        
    return {
        'documents': docs,
        'questions': qs
    }


def save_sample_data(output_dir: str = 'data'):
    """Save sample data to files"""
    import json
    import os

    os.makedirs(output_dir, exist_ok=True)

    for doc_id, text in SAMPLE_DOCUMENTS.items():
        with open(f'{output_dir}/{doc_id}.txt', 'w') as f:
            f.write(text)

    with open(f'{output_dir}/questions.json', 'w') as f:
        json.dump(SAMPLE_QUESTIONS, f, indent=2)

    print(f"[OK] Sample data saved to {output_dir}/")
    print(f"  - {len(SAMPLE_DOCUMENTS)} documents")
    print(f"  - {len(SAMPLE_QUESTIONS)} questions")


if __name__ == "__main__":
    save_sample_data()
