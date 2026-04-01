REFERRAL_MATRIX = {
    ('nv', 'Low Risk'): (
        'No Action Needed', 
        'This appears to be a common mole...monitor for changes in size, color, shape. Re-visit if lesion grows beyond 6mm.', 
        'GREEN'
    ),
    ('df', 'Low Risk'): (
        'Monitor Regularly', 
        'Likely a benign skin growth. Not dangerous. Recommend annual skin check.', 
        'GREEN'
    ),
    ('bkl', 'Low Risk'): (
        'Monitor — No Rush', 
        'Benign skin lesion. Not cancerous. Avoid sun exposure. Follow-up in 3 months if change observed.', 
        'GREEN'
    ),
    ('vasc', 'Refer Soon'): (
        'Refer Within 1 Week', 
        'Vascular lesion. Fill referral form, send to PHC doctor within 7 days. Do not apply pressure.', 
        'YELLOW'
    ),
    ('akiec', 'Refer Soon'): (
        'Refer Within 3 Days — Pre-cancerous', 
        'Possible pre-cancerous lesion. FILL REFERRAL SLIP NOW. Schedule district hospital appointment.', 
        'YELLOW'
    ),
    ('bcc', 'Refer Immediately'): (
        'URGENT — Refer Within 24 Hours', 
        'Possible Basal Cell Carcinoma. Complete emergency referral form. District hospital within 24 hours.', 
        'RED'
    ),
    ('mel', 'Refer Immediately'): (
        'EMERGENCY — Refer TODAY', 
        'High-risk melanoma signal. Medical emergency. Call district hospital NOW. Transport patient today.', 
        'RED'
    )
}

REFERRAL_HINDI = {
    ('nv', 'Low Risk'): (
        'कोई कार्रवाई आवश्यक नहीं', 
        'यह एक सामान्य तिल प्रतीत होता है... इसके आकार, रंग और रूप में बदलाव की निगरानी करें। यदि घाव 6 मिमी से बड़ा हो जाता है तो पुनः दिखाएं।', 
        'GREEN'
    ),
    ('df', 'Low Risk'): (
        'नियमित निगरानी करें', 
        'यह एक सौम्य त्वचा की वृद्धि है। खतरनाक नहीं है। वार्षिक त्वचा जांच की सिफारिश की जाती है।', 
        'GREEN'
    ),
    ('bkl', 'Low Risk'): (
        'निगरानी करें — कोई जल्दी नहीं', 
        'सौम्य त्वचा का घाव। कैंसर रहित है। धूप से बचें। यदि कोई बदलाव दिखाई दे तो 3 महीने में फॉलो-अप करें।', 
        'GREEN'
    ),
    ('vasc', 'Refer Soon'): (
        '1 सप्ताह के भीतर रेफर करें', 
        'संवहनी घाव। रेफरल फॉर्म भरें, 7 दिनों के भीतर पीएचसी डॉक्टर के पास भेजें। दबाव लागू न करें।', 
        'YELLOW'
    ),
    ('akiec', 'Refer Soon'): (
        '3 दिन के भीतर रेफर करें — पूर्व कैंसर', 
        'संभावित पूर्व कैंसर घाव। अभी रेफरल पर्ची भरें। जिला अस्पताल में अपॉइंटमेंट लें।', 
        'YELLOW'
    ),
    ('bcc', 'Refer Immediately'): (
        'अति आवश्यक — 24 घंटे के भीतर रेफर करें', 
        'संभावित बेसल सेल कार्सिनोमा। आपातकालीन रेफरल फॉर्म पूरा करें। 24 घंटे के भीतर जिला अस्पताल रेफर करें।', 
        'RED'
    ),
    ('mel', 'Refer Immediately'): (
        'आपातकाल — आज ही रेफर करें', 
        'उच्च जोखिम वाले मेलेनोमा का संकेत। चिकित्सा आपातकाल। जिला अस्पताल को अभी बुलाएं। मरीज को आज ही अस्पताल पहुंचाएं।', 
        'RED'
    )
}

SEVERITY_MAP = {
    'nv': 'Low Risk',
    'df': 'Low Risk',
    'bkl': 'Low Risk',
    'vasc': 'Refer Soon',
    'akiec': 'Refer Soon',
    'bcc': 'Refer Immediately',
    'mel': 'Refer Immediately'
}

def generate_referral(condition_code, severity_tier, confidence, top3_conditions, lang='en'):
    matrix = REFERRAL_HINDI if lang == 'hi' else REFERRAL_MATRIX
    
    key = (condition_code, severity_tier)
    
    if key in matrix:
        action_title, instruction, urgency_color = matrix[key]
    else:
        # Fallback if key is somehow mismatched 
        action_title = 'Unspecified Action'
        instruction = 'Monitor patient and consult doctor if unsure.'
        urgency_color = 'GRAY'
        
    # Override for low confidence
    if confidence < 0.60:
        if lang == 'hi':
            action_title = 'जल्द रेफर करें - कम विश्वास परिणाम'
        else:
            action_title = 'Refer Soon — Low Confidence Result'
        urgency_color = 'YELLOW'
        
    return {
        'condition': condition_code,
        'severity': severity_tier,
        'urgency_color': urgency_color,
        'action_title': action_title,
        'instruction': instruction,
        'confidence_pct': round(confidence * 100, 1),
        'top3': top3_conditions
    }

def get_severity_from_condition(condition_code):
    return SEVERITY_MAP.get(condition_code, 'Unknown')
