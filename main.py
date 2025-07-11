import streamlit as st
import json
import re
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import pandas as pd
from datetime import datetime

# LangChain imports
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.schema import BaseOutputParser
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

# Data Models
class DataType(Enum):
    BOOL = "BOOL"
    INT = "INT"
    REAL = "REAL"
    DINT = "DINT"
    STRING = "STRING"
    WORD = "WORD"
    DWORD = "DWORD"

class AccessType(Enum):
    READ_ONLY = "READ_ONLY"
    WRITE_ONLY = "WRITE_ONLY"
    READ_WRITE = "READ_WRITE"

@dataclass
class HMIElement:
    name: str
    element_type: str
    properties: Dict[str, Any]
    position: Tuple[int, int]
    size: Tuple[int, int]

@dataclass
class PLCTag:
    name: str
    data_type: DataType
    address: str
    access_type: AccessType
    description: str
    initial_value: Any = None

class PLCCodeOutput(BaseModel):
    data_blocks: List[Dict[str, Any]] = Field(description="Generated PLC data blocks")
    communication_tags: List[Dict[str, Any]] = Field(description="Communication tag definitions")
    status_logic: str = Field(description="Status reporting logic code")
    optimization_notes: List[str] = Field(description="Code optimization suggestions")

class PLCCodeParser(BaseOutputParser):
    def parse(self, text: str) -> PLCCodeOutput:
        try:
            # Extract structured data from LLM response
            data_blocks = self._extract_data_blocks(text)
            comm_tags = self._extract_communication_tags(text)
            status_logic = self._extract_status_logic(text)
            optimization_notes = self._extract_optimization_notes(text)
            
            return PLCCodeOutput(
                data_blocks=data_blocks,
                communication_tags=comm_tags,
                status_logic=status_logic,
                optimization_notes=optimization_notes
            )
        except Exception as e:
            return PLCCodeOutput(
                data_blocks=[],
                communication_tags=[],
                status_logic="// Error generating status logic",
                optimization_notes=[f"Error: {str(e)}"]
            )
    
    def _extract_data_blocks(self, text: str) -> List[Dict[str, Any]]:
        # Extract data block definitions from generated text
        blocks = []
        pattern = r'DATA_BLOCK\s+(\w+).*?END_DATA_BLOCK'
        matches = re.findall(pattern, text, re.DOTALL)
        for match in matches:
            blocks.append({"name": match, "type": "DB"})
        return blocks
    
    def _extract_communication_tags(self, text: str) -> List[Dict[str, Any]]:
        # Extract communication tag definitions
        tags = []
        pattern = r'(\w+)\s*:\s*(\w+)\s*:=\s*([^;]+);'
        matches = re.findall(pattern, text)
        for name, data_type, address in matches:
            tags.append({
                "name": name,
                "data_type": data_type,
                "address": address
            })
        return tags
    
    def _extract_status_logic(self, text: str) -> str:
        # Extract status logic code
        pattern = r'// STATUS LOGIC START(.*?)// STATUS LOGIC END'
        match = re.search(pattern, text, re.DOTALL)
        return match.group(1).strip() if match else ""
    
    def _extract_optimization_notes(self, text: str) -> List[str]:
        # Extract optimization suggestions
        pattern = r'// OPTIMIZATION: (.*?)(?=\n|$)'
        matches = re.findall(pattern, text)
        return matches

class HMIPLCIntegrationAI:
    def __init__(self, api_key: str = None):
        self.llm = OpenAI(temperature=0.3, openai_api_key=api_key) if api_key else None
        self.parser = PLCCodeParser()
        self.setup_prompts()
    
    def setup_prompts(self):
        self.code_generation_prompt = PromptTemplate(
            input_variables=["hmi_elements", "plc_type", "communication_protocol"],
            template="""
            You are an expert PLC programmer and HMI integration specialist.
            
            Generate optimized PLC communication code based on the following HMI screen design:
            
            HMI Elements:
            {hmi_elements}
            
            PLC Type: {plc_type}
            Communication Protocol: {communication_protocol}
            
            Please generate:
            1. Data blocks with appropriate structure
            2. Communication tags with proper addressing
            3. Status reporting logic
            4. Optimization recommendations
            
            Format your response with clear sections:
            - DATA_BLOCK definitions
            - Communication tag mappings
            - Status logic code
            - Optimization notes
            
            Ensure the code is production-ready and follows best practices.
            """
        )
        
        self.chain = LLMChain(
            llm=self.llm,
            prompt=self.code_generation_prompt,
            output_parser=self.parser
        ) if self.llm else None
    
    def analyze_hmi_design(self, hmi_elements: List[HMIElement]) -> Dict[str, Any]:
        """Analyze HMI design and extract communication requirements"""
        analysis = {
            "input_elements": [],
            "output_elements": [],
            "bidirectional_elements": [],
            "status_elements": [],
            "data_requirements": {}
        }
        
        for element in hmi_elements:
            element_type = element.element_type.lower()
            
            if element_type in ['button', 'switch', 'setpoint']:
                analysis["input_elements"].append(element)
            elif element_type in ['indicator', 'gauge', 'display']:
                analysis["output_elements"].append(element)
            elif element_type in ['slider', 'numericinput']:
                analysis["bidirectional_elements"].append(element)
            elif element_type in ['alarm', 'status', 'led']:
                analysis["status_elements"].append(element)
        
        return analysis
    
    def generate_plc_tags(self, analysis: Dict[str, Any]) -> List[PLCTag]:
        """Generate PLC tags based on HMI analysis"""
        tags = []
        
        # Generate tags for input elements
        for element in analysis["input_elements"]:
            tag = PLCTag(
                name=f"HMI_{element.name}",
                data_type=DataType.BOOL if element.element_type.lower() == 'button' else DataType.INT,
                address=f"DB1.DBX{len(tags)}.0",
                access_type=AccessType.READ_ONLY,
                description=f"HMI input from {element.name}"
            )
            tags.append(tag)
        
        # Generate tags for output elements
        for element in analysis["output_elements"]:
            data_type = DataType.REAL if 'gauge' in element.element_type.lower() else DataType.BOOL
            tag = PLCTag(
                name=f"TO_HMI_{element.name}",
                data_type=data_type,
                address=f"DB2.DBX{len(tags)}.0",
                access_type=AccessType.WRITE_ONLY,
                description=f"PLC output to HMI {element.name}"
            )
            tags.append(tag)
        
        return tags
    
    def generate_communication_code(self, hmi_elements: List[HMIElement], 
                                   plc_type: str, protocol: str) -> PLCCodeOutput:
        """Generate complete PLC communication code"""
        if not self.chain:
            # Fallback generation without AI
            return self._generate_fallback_code(hmi_elements, plc_type, protocol)
        
        elements_text = self._format_elements_for_prompt(hmi_elements)
        
        try:
            result = self.chain.run(
                hmi_elements=elements_text,
                plc_type=plc_type,
                communication_protocol=protocol
            )
            return result
        except Exception as e:
            st.error(f"AI generation failed: {str(e)}")
            return self._generate_fallback_code(hmi_elements, plc_type, protocol)
    
    def _format_elements_for_prompt(self, hmi_elements: List[HMIElement]) -> str:
        """Format HMI elements for LLM prompt"""
        formatted = []
        for element in hmi_elements:
            formatted.append(f"- {element.name}: {element.element_type} at {element.position}")
        return "\n".join(formatted)
    
    def _generate_fallback_code(self, hmi_elements: List[HMIElement], 
                               plc_type: str, protocol: str) -> PLCCodeOutput:
        """Generate basic code without AI"""
        analysis = self.analyze_hmi_design(hmi_elements)
        tags = self.generate_plc_tags(analysis)
        
        # Generate basic data blocks
        data_blocks = [
            {"name": "HMI_INPUT_DB", "type": "DB", "number": 1},
            {"name": "HMI_OUTPUT_DB", "type": "DB", "number": 2}
        ]
        
        # Generate communication tags
        comm_tags = [
            {
                "name": tag.name,
                "data_type": tag.data_type.value,
                "address": tag.address,
                "access": tag.access_type.value
            }
            for tag in tags
        ]
        
        # Generate basic status logic
        status_logic = """
// STATUS LOGIC START
// HMI Communication Status
HMI_COMM_STATUS := TRUE;
HMI_COMM_ERROR := FALSE;

// Heartbeat logic
IF HMI_HEARTBEAT_COUNTER >= 1000 THEN
    HMI_HEARTBEAT_COUNTER := 0;
    HMI_HEARTBEAT := NOT HMI_HEARTBEAT;
END_IF;
HMI_HEARTBEAT_COUNTER := HMI_HEARTBEAT_COUNTER + 1;
// STATUS LOGIC END
        """
        
        optimization_notes = [
            "Consider using structured data types for complex HMI elements",
            "Implement proper error handling for communication timeouts",
            "Use indirect addressing for scalable tag management"
        ]
        
        return PLCCodeOutput(
            data_blocks=data_blocks,
            communication_tags=comm_tags,
            status_logic=status_logic,
            optimization_notes=optimization_notes
        )

def create_sample_hmi_elements() -> List[HMIElement]:
    """Create sample HMI elements for testing"""
    return [
        HMIElement("StartButton", "Button", {"color": "green"}, (100, 100), (80, 40)),
        HMIElement("StopButton", "Button", {"color": "red"}, (200, 100), (80, 40)),
        HMIElement("TemperatureDisplay", "Display", {"format": "0.0"}, (100, 200), (120, 30)),
        HMIElement("PressureGauge", "Gauge", {"min": 0, "max": 100}, (300, 150), (150, 150)),
        HMIElement("SpeedSetpoint", "NumericInput", {"min": 0, "max": 1000}, (100, 300), (120, 30)),
        HMIElement("AlarmIndicator", "LED", {"color": "red"}, (400, 100), (30, 30)),
        HMIElement("StatusDisplay", "Status", {"states": ["OFF", "ON", "ERROR"]}, (100, 400), (200, 30))
    ]

def main():
    st.set_page_config(
        page_title="HMI-PLC Integration AI",
        page_icon="🏭",
        layout="wide"
    )
    
    st.title("🏭 Intelligent HMI-PLC Integration System")
    st.markdown("Automatically generate PLC communication code based on HMI screen designs")
    
    # Sidebar configuration
    st.sidebar.header("Configuration")
    
    # API Key input
    api_key = st.sidebar.text_input("OpenAI API Key (optional)", type="password")
    
    # PLC Configuration
    plc_type = st.sidebar.selectbox(
        "PLC Type",
        ["Siemens S7-1200", "Siemens S7-1500", "Allen-Bradley CompactLogix", 
         "Schneider Modicon M580", "Omron NJ Series"]
    )
    
    communication_protocol = st.sidebar.selectbox(
        "Communication Protocol",
        ["OPC UA", "Modbus TCP", "Ethernet/IP", "Profinet"]
    )
    
    # Initialize AI system
    ai_system = HMIPLCIntegrationAI(api_key)
    
    # Main interface
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("HMI Screen Design")
        
        # Option to use sample data or upload
        use_sample = st.checkbox("Use Sample HMI Design", value=True)
        
        if use_sample:
            hmi_elements = create_sample_hmi_elements()
            st.success(f"Loaded {len(hmi_elements)} sample HMI elements")
        else:
            st.info("HMI design upload functionality would be implemented here")
            hmi_elements = []
        
        # Display HMI elements
        if hmi_elements:
            st.subheader("HMI Elements")
            elements_df = pd.DataFrame([
                {
                    "Name": elem.name,
                    "Type": elem.element_type,
                    "Position": f"{elem.position[0]}, {elem.position[1]}",
                    "Size": f"{elem.size[0]} x {elem.size[1]}"
                }
                for elem in hmi_elements
            ])
            st.dataframe(elements_df, use_container_width=True)
    
    with col2:
        st.header("PLC Code Generation")
        
        if hmi_elements and st.button("Generate PLC Code", type="primary"):
            with st.spinner("Analyzing HMI design and generating PLC code..."):
                # Analyze HMI design
                analysis = ai_system.analyze_hmi_design(hmi_elements)
                
                # Generate PLC code
                plc_code = ai_system.generate_communication_code(
                    hmi_elements, plc_type, communication_protocol
                )
                
                # Display results
                st.success("PLC code generated successfully!")
                
                # Analysis results
                st.subheader("HMI Analysis")
                col_a, col_b = st.columns(2)
                
                with col_a:
                    st.metric("Input Elements", len(analysis["input_elements"]))
                    st.metric("Output Elements", len(analysis["output_elements"]))
                
                with col_b:
                    st.metric("Bidirectional Elements", len(analysis["bidirectional_elements"]))
                    st.metric("Status Elements", len(analysis["status_elements"]))
    
    # Results section
    if 'plc_code' in locals():
        st.header("Generated PLC Code")
        
        # Tabs for different code sections
        tab1, tab2, tab3, tab4 = st.tabs([
            "Data Blocks", "Communication Tags", "Status Logic", "Optimization Notes"
        ])
        
        with tab1:
            st.subheader("Data Blocks")
            if plc_code.data_blocks:
                for db in plc_code.data_blocks:
                    st.code(f"DATA_BLOCK {db['name']}\nTYPE: {db['type']}\nEND_DATA_BLOCK", language="text")
            else:
                st.info("No data blocks generated")
        
        with tab2:
            st.subheader("Communication Tags")
            if plc_code.communication_tags:
                tags_df = pd.DataFrame(plc_code.communication_tags)
                st.dataframe(tags_df, use_container_width=True)
                
                # Generate tag code
                tag_code = "\n".join([
                    f"{tag['name']} : {tag['data_type']} := {tag['address']};"
                    for tag in plc_code.communication_tags
                ])
                st.code(tag_code, language="text")
            else:
                st.info("No communication tags generated")
        
        with tab3:
            st.subheader("Status Logic")
            if plc_code.status_logic:
                st.code(plc_code.status_logic, language="text")
            else:
                st.info("No status logic generated")
        
        with tab4:
            st.subheader("Optimization Notes")
            if plc_code.optimization_notes:
                for note in plc_code.optimization_notes:
                    st.info(f"💡 {note}")
            else:
                st.info("No optimization notes available")
        
        # Download generated code
        st.subheader("Export Code")
        
        # Prepare complete code for download
        complete_code = f"""
// Generated PLC Code for HMI Integration
// PLC Type: {plc_type}
// Communication Protocol: {communication_protocol}
// Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

// DATA BLOCKS
{chr(10).join([f"DATA_BLOCK {db['name']}" for db in plc_code.data_blocks])}

// COMMUNICATION TAGS
{chr(10).join([f"{tag['name']} : {tag['data_type']} := {tag['address']};" for tag in plc_code.communication_tags])}

// STATUS LOGIC
{plc_code.status_logic}

// OPTIMIZATION NOTES
{chr(10).join([f"// {note}" for note in plc_code.optimization_notes])}
        """
        
        st.download_button(
            label="Download Complete PLC Code",
            data=complete_code,
            file_name=f"hmi_plc_code_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            mime="text/plain"
        )
    
    # Footer
    st.markdown("---")
    st.markdown(
        "Built with ❤️ using Streamlit and LangChain | "
        "For production use, ensure proper testing and validation of generated code"
    )

if __name__ == "__main__":
    main()