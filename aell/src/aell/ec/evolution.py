import re
import time
from prepareAEL import GetPrompts
from ..llm.api_local_llm import LocalLLM
from ..llm.interface_LLM import get_response

class Evolution:

    def __init__(self, debug_mode, object, ob, **kwargs):
        # -------------------- RZ: use local LLM --------------------
        assert 'use_local_llm' in kwargs
        assert 'url' in kwargs
        self._use_local_llm = kwargs.get('use_local_llm')
        self._url = kwargs.get('url')
        # -----------------------------------------------------------

        # -------------------- RZ: use local LLM --------------------
        if self._use_local_llm:
            self.interface_llm = LocalLLM(self._url)

        self.object = object
        self.debug_mode = debug_mode
        self.ob = ob

        
    def get_prompt_e1(self,indivs):
        prompt_indiv = ""
        for i in range(len(indivs)):
            prompt_indiv = prompt_indiv + "No." + str(i + 1) + " image caption are: \n" + indivs[i]['code'] + "\n"
        if self.object:
            pass
        else:
            prompt_content = "I have "+str(len(indivs))+" existing image captions as follows: \n"+prompt_indiv+"The sentence format of these captions is a <picture/photo/watercolor/sketch> of <number> <color> <object> <appearance> in the style of <style>. <They/It/He/She> <gesture/is/are> on the <background describe> in the <location> on a <weather> day, <action/detailed description>, <environment description>. Please help me create only a image description sentence according to the sentence format I give above and it is completely different from the given captions. You can add or change any descriptions, such as object, style, weather, location, action, environment and background. Do not give any additional explanations or irrelavent texts."
        return prompt_content
    
    def get_prompt_m1(self,indiv1):
        if self.object:
            pass
        else:
            prompt_content = "I have one image caption as follows. Caption:\n"+indiv1['code']+"\nPlease help me write only a image caption that is a variation of the original caption. You can modify or add some decorative or descriptive sentences/words. You can also be inspired by it to add some other elements. You can also change the apperance, style, weather, light and environment.  \nThe sentence structure of caption is a <picture/photo/watercolor/sketch> of <number> <color> <object> <appearance> in the style of <style>. <They/It/He/She> <gesture/is/are> on the <background describe> in the <location> on a <weather> day, <action/detailed description>, <environment description>. Do not give any additional explanations or irrelavent texts."
        return prompt_content


    def _get_alg(self,prompt_content):

        response = get_response(prompt_content)
        prompt = "Please ensure that the sentence: "+response+"is a <picture/photo/watercolor/sketch> of <number> <color> <object><appearance>in the style of<style>. <They/It/He/She> <gesture/is/are> on the <background describe> in the <location> on a <weather> day, <action/detailed description> ,<environment description>. If not, please modify it and only return the modified sentence. Do not give any additional explanations or irrelavent texts."
        response = get_response(prompt)
        algorithm = ""
        sentence = response
        if '\n' in sentence:
            sentence = sentence.replace('\n', '')
        if 'Caption' in sentence or 'caption' in sentence:
            sentence = sentence.replace('Caption', '').replace('caption', '')

        # Step 3: 选取冒号后的内容
        if ':' in sentence:
            sentence = sentence.split(':', 1)[1]

        # Step 4: 去掉句子两边的空格
        sentence = sentence.strip()
        #print(code_all)
        return [sentence, algorithm]


    def i1(self):

        prompt_content = self.get_prompt_i1()

        if self.debug_mode:
            print("\n >>> check prompt for creating algorithm using [ i1 ] : \n", prompt_content )
            print(">>> Press 'Enter' to continue")
            input()
      
        [code_all, algorithm] = self._get_alg(prompt_content)

        if self.debug_mode:
            print("\n >>> check designed algorithm: \n", algorithm)
            print("\n >>> check designed code: \n", code_all)
            print(">>> Press 'Enter' to continue")
            input()

        return [code_all, algorithm]
    
    def e1(self,parents):
        prompt_content = self.get_prompt_e1(parents)

        if self.debug_mode:
            print("\n >>> check prompt for creating algorithm using [ e1 ] : \n", prompt_content )
            print(">>> Press 'Enter' to continue")
            input()
      
        [code_all, algorithm] = self._get_alg(prompt_content)

        if self.debug_mode:
            print("\n >>> check designed algorithm: \n", algorithm)
            print("\n >>> check designed code: \n", code_all)
            print(">>> Press 'Enter' to continue")
            input()

        return [code_all, algorithm]
    
    def e2(self,parents):
      
        prompt_content = self.get_prompt_e2(parents)

        if self.debug_mode:
            print("\n >>> check prompt for creating algorithm using [ e2 ] : \n", prompt_content )
            print(">>> Press 'Enter' to continue")
            input()
      
        [code_all, algorithm] = self._get_alg(prompt_content)

        if self.debug_mode:
            print("\n >>> check designed algorithm: \n", algorithm)
            print("\n >>> check designed code: \n", code_all)
            print(">>> Press 'Enter' to continue")
            input()

        return [code_all, algorithm]
    
    def m1(self,parents):
      
        prompt_content = self.get_prompt_m1(parents)

        if self.debug_mode:
            print("\n >>> check prompt for creating algorithm using [ m1 ] : \n", prompt_content )
            print(">>> Press 'Enter' to continue")
            input()
      
        [code_all, algorithm] = self._get_alg(prompt_content)

        if self.debug_mode:
            print("\n >>> check designed algorithm: \n", algorithm)
            print("\n >>> check designed code: \n", code_all)
            print(">>> Press 'Enter' to continue")
            input()

        return [code_all, algorithm]
    
    def m2(self,parents):
      
        prompt_content = self.get_prompt_m2(parents)

        if self.debug_mode:
            print("\n >>> check prompt for creating algorithm using [ m2 ] : \n", prompt_content )
            print(">>> Press 'Enter' to continue")
            input()
      
        [code_all, algorithm] = self._get_alg(prompt_content)

        if self.debug_mode:
            print("\n >>> check designed algorithm: \n", algorithm)
            print("\n >>> check designed code: \n", code_all)
            print(">>> Press 'Enter' to continue")
            input()

        return [code_all, algorithm]