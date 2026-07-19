TEST_CASES = [
    # ============================================================
    # SUPPORT
    # ============================================================

    {
        "query": "What is the refund policy for standard customers?",
        "tenant_id": "tenant_1",
        "department": "support",
        "relevant_sources": [
            "refund_policy_standard_customers.pdf",
        ],
    },
    {
        "query": "What is the refund policy for enterprise customers?",
        "tenant_id": "tenant_1",
        "department": "support",
        "relevant_sources": [
            "refund_policy_enterprise_customers.pdf",
        ],
    },

    # ============================================================
    # HR
    # ============================================================

    {
        "query": "How do employees enroll in benefits?",
        "tenant_id": "tenant_1",
        "department": "hr",
        "relevant_sources": [
            "benefits_enrollment_guide.pdf",
        ],
    },
    {
        "query": "What is the annual leave policy?",
        "tenant_id": "tenant_1",
        "department": "hr",
        "relevant_sources": [
            "annual_leave_policy.pdf",
        ],
    },
    {
        "query": "What is the sick leave policy?",
        "tenant_id": "tenant_1",
        "department": "hr",
        "relevant_sources": [
            "sick_leave_policy.pdf",
        ],
    },
    {
        "query": "What is the maternity and parental leave policy?",
        "tenant_id": "tenant_1",
        "department": "hr",
        "relevant_sources": [
            "maternity_parental_leave_policy.pdf",
        ],
    },
    {
        "query": "What is the regional leave policy for India?",
        "tenant_id": "tenant_1",
        "department": "hr",
        "relevant_sources": [
            "regional_leave_policy_india.pdf",
        ],
    },

    # ============================================================
    # ENGINEERING
    # ============================================================

    {
        "query": "Which API endpoint handles document ingestion?",
        "tenant_id": "tenant_1",
        "department": "engineering",
        "relevant_sources": [
            "document_ingestion_api_reference.yaml",
        ],
    },
    {
        "query": "What are the API rate limiting rules?",
        "tenant_id": "tenant_1",
        "department": "engineering",
        "relevant_sources": [
            "api_rate_limiting_policy.pdf",
        ],
    },
    {
        "query": "How does the CI/CD pipeline work?",
        "tenant_id": "tenant_1",
        "department": "engineering",
        "relevant_sources": [
            "cicd_pipeline_documentation.pdf",
        ],
    },
    {
        "query": "How is the payment API authenticated?",
        "tenant_id": "tenant_1",
        "department": "engineering",
        "relevant_sources": [
            "payment_api_authentication_guide.pdf",
        ],
    },
    {
        "query": "What are the service uptime metrics for 2025?",
        "tenant_id": "tenant_1",
        "department": "engineering",
        "relevant_sources": [
            "service_uptime_metrics_2025.csv",
        ],
    },

    # ============================================================
    # FINANCE
    # ============================================================

    {
        "query": "What does ARR mean?",
        "tenant_id": "tenant_1",
        "department": "finance",
        "relevant_sources": [
            "finance_glossary.json",
        ],
    },
    {
        "query": "What does ASC 606 mean?",
        "tenant_id": "tenant_1",
        "department": "finance",
        "relevant_sources": [
            "revenue_recognition_policy_asc606.pdf",
        ],
    },

    # ============================================================
    # SUPPORT
    # ============================================================

    {
        "query": "How do I reset my password?",
        "tenant_id": "tenant_1",
        "department": "support",
        "relevant_sources": [
            "faq_common_product_questions.md",
        ],
    },
    {
        "query": "What are the common product questions?",
        "tenant_id": "tenant_1",
        "department": "support",
        "relevant_sources": [
            "faq_common_product_questions.md",
        ],
    },

    # ============================================================
    # ENGINEERING
    # ============================================================

    {
        "query": "What is the document ingestion endpoint rate limit?",
        "tenant_id": "tenant_1",
        "department": "engineering",
        "relevant_sources": [
            "document_ingestion_api_reference.yaml",
            "api_rate_limiting_policy.pdf",
        ],
    },
    {
        "query": "Who owns the Query Service in the system architecture?",
        "tenant_id": "tenant_1",
        "department": "engineering",
        "relevant_sources": [
            "system_architecture_overview.pdf",
        ],
    },

    # ============================================================
    # HR
    # ============================================================

    {
        "query": "What steps are involved in benefits enrollment?",
        "tenant_id": "tenant_1",
        "department": "hr",
        "relevant_sources": [
            "benefits_enrollment_guide.pdf",
        ],
    },

    # ============================================================
    # SUPPORT
    # ============================================================

    {
        "query": "What are the refund rules for Standard-tier customers?",
        "tenant_id": "tenant_1",
        "department": "support",
        "relevant_sources": [
            "refund_policy_standard_customers.pdf",
        ],
    },

    # ============================================================
    # FINANCE
    # ============================================================

    {
        "query": "What expenses can employees claim for reimbursement?",
        "tenant_id": "tenant_1",
        "department": "finance",
        "relevant_sources": [
            "employee_expense_reimbursement_policy.pdf",
        ],
    },
    {
        "query": "What is the company's revenue recognition policy?",
        "tenant_id": "tenant_1",
        "department": "finance",
        "relevant_sources": [
            "revenue_recognition_policy_asc606.pdf",
        ],
    },
    {
        "query": "How did revenue vary by geographic region?",
        "tenant_id": "tenant_1",
        "department": "finance",
        "relevant_sources": [
            "monthly_revenue_by_region.csv",
        ],
    },
    {
        "query": "What were the financial results for Q1 2025?",
        "tenant_id": "tenant_1",
        "department": "finance",
        "relevant_sources": [
            "quarterly_earnings_summary_q1_2025.pdf",
        ],
    },
    {
        "query": "What are the approval limits for the finance team?",
        "tenant_id": "tenant_1",
        "department": "finance",
        "relevant_sources": [
            "finance_team_directory_and_approval_limits.json",
        ],
    },
    {
        "query": "How are cost centers defined within the company?",
        "tenant_id": "tenant_1",
        "department": "finance",
        "relevant_sources": [
            "cost_center_definitions.docx",
        ],
    },
    {
        "query": "What is the capital expenditure plan for 2025?",
        "tenant_id": "tenant_1",
        "department": "finance",
        "relevant_sources": [
            "capital_expenditure_plan_2025.pdf",
        ],
    },
    {
        "query": "How does the procurement approval process work?",
        "tenant_id": "tenant_1",
        "department": "finance",
        "relevant_sources": [
            "procurement_approval_workflow.html",
        ],
    },

    # ============================================================
    # HR
    # ============================================================

    {
        "query": "What are the rules for working remotely?",
        "tenant_id": "tenant_1",
        "department": "hr",
        "relevant_sources": [
            "remote_work_policy.html",
        ],
    },
    {
        "query": "How does the employee performance review process work?",
        "tenant_id": "tenant_1",
        "department": "hr",
        "relevant_sources": [
            "performance_review_process.html",
        ],
    },
    {
        "query": "What should a new employee do during onboarding?",
        "tenant_id": "tenant_1",
        "department": "hr",
        "relevant_sources": [
            "employee_onboarding_handbook.pdf",
        ],
    },
    {
        "query": "How can an employee submit a workplace grievance?",
        "tenant_id": "tenant_1",
        "department": "hr",
        "relevant_sources": [
            "workplace_grievance_policy.pdf",
        ],
    },
    {
        "query": "Who should employees contact in the HR department?",
        "tenant_id": "tenant_1",
        "department": "hr",
        "relevant_sources": [
            "hr_contact_directory.json",
        ],
    },
    {
        "query": "What holidays are observed by the company in 2025?",
        "tenant_id": "tenant_1",
        "department": "hr",
        "relevant_sources": [
            "company_holiday_calendar_2025.csv",
        ],
    },

    # ============================================================
    # ENGINEERING
    # ============================================================

    {
        "query": "What standards should engineers follow when reviewing code?",
        "tenant_id": "tenant_1",
        "department": "engineering",
        "relevant_sources": [
            "code_review_standards.docx",
        ],
    },
    {
        "query": "How should Kubernetes deployments be performed?",
        "tenant_id": "tenant_1",
        "department": "engineering",
        "relevant_sources": [
            "kubernetes_deployment_runbook.pdf",
        ],
    },
    {
        "query": "What is the architecture of the company's microservices?",
        "tenant_id": "tenant_1",
        "department": "engineering",
        "relevant_sources": [
            "system_architecture_overview.pdf",
        ],
    },
    {
        "query": "What should engineers do after a production incident?",
        "tenant_id": "tenant_1",
        "department": "engineering",
        "relevant_sources": [
            "post_incident_review_process.pdf",
        ],
    },
    {
        "query": "Which technologies are used by the engineering teams?",
        "tenant_id": "tenant_1",
        "department": "engineering",
        "relevant_sources": [
            "engineering_team_directory_and_tech_stack.json",
        ],
    },
    {
        "query": "What caused the database connection failure incident?",
        "tenant_id": "tenant_1",
        "department": "engineering",
        "relevant_sources": [
            "database_connection_failure_incident_report.pdf",
        ],
    },

    # ============================================================
    # SUPPORT
    # ============================================================

    {
        "query": "How should support tickets be escalated?",
        "tenant_id": "tenant_1",
        "department": "support",
        "relevant_sources": [
            "support_ticket_escalation_policy.pdf",
        ],
    },
    {
        "query": "What service levels are provided for different support tiers?",
        "tenant_id": "tenant_1",
        "department": "support",
        "relevant_sources": [
            "support_service_level_agreement.pdf",
        ],
    },
    {
        "query": "What known product issues have documented workarounds?",
        "tenant_id": "tenant_1",
        "department": "support",
        "relevant_sources": [
            "known_product_issues_and_workarounds.pdf",
        ],
    },
    {
        "query": "How should support agents respond to customers in live chat?",
        "tenant_id": "tenant_1",
        "department": "support",
        "relevant_sources": [
            "live_chat_support_guidelines.pdf",
        ],
    },
    {
        "query": "How should support staff troubleshoot the core platform?",
        "tenant_id": "tenant_1",
        "department": "support",
        "relevant_sources": [
            "core_platform_troubleshooting_guide.pdf",
        ],
    },

    # ============================================================
    # LEGAL
    # ============================================================

    {
        "query": "What are the GDPR requirements for processing personal data?",
        "tenant_id": "tenant_1",
        "department": "legal",
        "relevant_sources": [
            "data_processing_addendum_gdpr.pdf",
            "regulatory_compliance_checklist_gdpr.html",
        ],
    },
    {
        "query": "What steps are required for GDPR regulatory compliance?",
        "tenant_id": "tenant_1",
        "department": "legal",
        "relevant_sources": [
            "regulatory_compliance_checklist_gdpr.html",
        ],
    },
    {
        "query": "What legal risks arise from data breach scenarios?",
        "tenant_id": "tenant_1",
        "department": "legal",
        "relevant_sources": [
            "data_breach_legal_risk_assessment.pdf",
        ],
    },
    {
        "query": "What are the termination clause differences between contracts?",
        "tenant_id": "tenant_1",
        "department": "legal",
        "relevant_sources": [
            "contract_termination_clause_comparison.pdf",
        ],
    },
    {
        "query": "What confidentiality obligations apply to vendors?",
        "tenant_id": "tenant_1",
        "department": "legal",
        "relevant_sources": [
            "vendor_confidentiality_agreement.pdf",
        ],
    },
]