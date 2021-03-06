"""
This file is part of HornMorpho, which is part of the PLoGS project.

    <http://homes.soic.indiana.edu/gasser/plogs.html>

    Copyleft 2020.
    PLoGS and Michael Gasser <gasser@indiana.edu>.

    morfo is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    morfo is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with morfo.  If not, see <http://www.gnu.org/licenses/>.
--------------------------------------------------------------------
Author: Michael Gasser <gasser@indiana.edu>

Functions common to some or all Ethio-Eritrean Semitic languages.
"""

#from . import language
from .geez import *
from .utils import segment, allcombs
#from .rule import *

class EES:

    # later extend to include preps and conjs from other languages
    ROM2GEEZ = \
    {'amh':
        {'sI': "ስ", 'lI': "ል", 'bI': "ብ", 'IskI': "እስክ", 'IndI': "እንድ",
         'm': "ም", 'Inji': "እንጂ", 'na': "ና", 'sa': "ሳ", 's': "ስ", 'ma': "ማ",
         'sIle': "ስለ", 'le': "ለ", 'Iyye': "እየ", 'Iske': "እስከ",
         'Inde': "እንደ", 'ke': "ከ", 'be': "በ", 'wede': "ወደ"
         }
    }

    def __init__(self):
        print("Creating EES language, {}".format(self))
        # EES pre-and post-processing: geezification, romanization,
        # handling of multi-word lexemes
        if not self.procroot:
            self.procroot = \
              lambda root, fs, pos: self.preproc_root(root, fs, pos)
        if not self.preproc:
            self.preproc = \
              lambda form: geez2sera(None, form, lang=self.abbrev,
                                     gemination=self.output_gemination)
        if not self.postproc:
            self.postproc = \
              lambda form: sera2geez(None, form, lang=self.abbrev,
                                     gemination=self.output_gemination)
        if not self.postpostproc:
            self.postpostproc = lambda form: form.replace('//', ' ')

    def preproc_root(self, root, fs, pos):
        """
        Preprocess root for generation.
        """
        if is_geez(root):
            if pos and pos.startswith('v'):
                # This is a verb root; we need the class
                anal = self.morphology[pos].anal(root, preproc=True)
                if anal:
                    root, ffss = anal[0]
                    cls = ffss.get('cls')
                    root, fs = \
                      self.dflt_procroot("{}:{}".format(root, cls), fs)
                else:
    #            print("Couldn't analyze {}".format(root))
                    return root, fs
            else:
                # Romanize noun roots
                root = geez2sera(None, root, self.abbrev)
                return root, fs
        else:
            root, fs = self.dflt_procroot(root, fs)
        return root, fs

    # def vb_get_citation(self, root, fs,
    #                     guess=False, vc_as=False):
    #     '''Return the canonical (prf, 3sm) form for the root and featstructs in
    #     featstruct fs.
    #     If at is True, use Amharic/Tigrinya features.
    #     If vc_as is True, preserve the voice and aspect of the original word.
    #     '''
    #     citation = ''
    #     if root == 'hlw':
    #         return "አለ"
    #     # Return root if no citation is found
    #     result = root
    #     # Unfreeze the feature structure
    #     fs = fs.unfreeze()
    #     fsa, fsv = fs.get('as'), fs.get('vc')
    #     # Update the feature structure to incorporate default (with or without vc and as)
    #     fs.update(self.language.morphology['v'].defaultFS)
    #     # For non-passive te- verbs, te- is citation form
    #     if fs.get('lexav') == True:
    #         # Lexical entry with explicit as, vc features
    #         fs.update({'as': fsa, "vc": fsv})
    #     elif fs.get('lexip') == True:
    #         # Lexical entry for as=it,vc=ps and as=it,vc=tr
    #         fs.update({'as': 'it', 'vc': 'ps'})
    #     elif fs.get('lexrp') == True:
    #         # Lexical entry for as=rc, vc=ps and as=rc,vc=tr
    #         fs.update({'as': 'rc', 'vc': 'ps'})
    #     elif fs.get('smp') == False:
    #         # No vc=smp form, vc=ps is base/citation form
    #         fs.update({'vc': 'ps'})
    #     # Refreeze the feature structure
    #     fs.freeze()
    #     # Find the first citation form compatible with the updated feature structure
    #     if ' ' in root:
    #         # This is a light verb, just generate the actual verb
    #         root_split = root.split()
    #         citation = self.language.morphology['v'].gen(root_split[-1], fs, from_dict=False,
    #                                            phon=True, postproc=False, guess=guess)
    #         if citation:
    #             result = ' '.join(root_split[:-1]) + ' ' + citation[0][0]
    #     else:
    # #        print("Generating citation from {}".format(fs.__repr__()))
    #         citation = self.language.morphology['v'].gen(root, fs, from_dict=False,
    #                                            phon=True, postproc=False, guess=guess)
    #         if citation:
    #             result = citation[0][0]
    #     if not citation:
    #         if not vc_as:
    #             # Verb may not occur in simplex form; try passive
    #             fs = fs.unfreeze()
    #             fs.update({'vc': 'ps'})
    #             fs.freeze()
    #             citation = self.language.morphology['v'].gen(root, fs, from_dict=False, guess=guess)
    #             if citation:
    #                 result = citation[0][0]
    #     return result
    #
    # def n_get_citation(self, root, fs, guess=False, vc_as=False):
    #     '''Return the canonical (prf, 3sm) form for the root and featstructs in featstruct set fss.
    #
    #     If vc_as is True, preserve the voice and aspect of the original word.
    #     '''
    #     if fs.get('v'):
    #         # It's a deverbal noun
    #         return vb_get_citation(root, fs, guess=guess, vc_as=vc_as)
    #     else:
    #         return None
    #
    # def simplify(self, word):
    #     """Simplify Amharic orthography."""
    #     word = word.replace("`", "'").replace('H', 'h').replace('^', '').replace('_', '')
    #     return word
    #
    # def orthographize(self, word):
    #     '''Convert phonological romanization to orthographic.'''
    #     word = word.replace('_', '').replace('I', '')
    #     return word
    #
    # def webfv(self, webdict, feature, value):
    #     if webdict != None:
    #         webdict[feature] = value
    #
    # def postpostproc_root(self, root, fs):
    #     """
    #     Convert root to root:class format, also changing internal
    #     HM root representation to an alternate conventional
    #     representation.
    #     """
    #     return "<{}:{}>".format(self.language.convert_root(root), fs['cls'])
    #
    # def postproc_nroot(self, root, fs):
    #     """
    #     Convert citation (lemma) to conventional phonetic representation.
    #     """
    #     if fs.get('pos') == 'n_dv':
    #         return "<{}:{}>".format(self.language.convert_root(root), fs['cls'])
    #     else:
    #         return "{}|{}".format(geezify(root), self.language.convert_phones(root))
    #
    # def postproc_word(self, word, ipa=False):
    #     """
    #     Convert output word to ortho|phon representation, also
    #     changing internal HM representation to an alternate
    #     conventional representation.
    #     """
    #     return "{}|{}".format(geezify(word), self.language.convert_phones(word))
    #
    # def load_anal(self, pos='v', lex=True, guess=False):
    #     if lex:
    #         self.language.morphology[pos].load_fst(True, verbose=True)
    #     if guess:
    #         self.language.morphology[pos].load_fst(True, guess=True, verbose=True)
    #
    # def load_gen(self, pos='v', lex=True, guess=False):
    #     if lex:
    #         self.language.morphology[pos].load_fst(True, generate=True, invert=True, verbose=True)
    #     if guess:
    #         self.language.morphology[pos].load_fst(True, generate=True, invert=True, guess=True, verbose=True)
    #
    # @staticmethod
    # def postproc_root(root):
    #     """Final adjustments to romanized root."""
    #     # Replace __ with space.
    #     if '//' in root:
    #         root = root.replace('//', ' ')
    #     return root
    #
    # @staticmethod
    # def roman2geez(value):
    #     """Convert a value (prep or conj) to geez."""
    #     return EES.ROM2GEEZ.get(value, value)
    #
    # def seg2string(self, segmentation, sep='-', geez=True, features=False,
    #                arules=False):
    #     """Convert a segmentation to a string, including features if features is True."""
    #     # The segmentation string is second in the list
    # #    print("Converting {} to string".format(segmentation))
    #     morphs, rootindex = self.language.seg2morphs(segmentation[1])
    #     # Root string and features
    #     root, rootfeats = morphs[rootindex]
    #     # Separate the consonants and template, and realize the root
    #     root = root2string(root)
    #     # Replace the root in the morphemes list
    #     morphs[rootindex] = root, rootfeats
    #     if geez:
    #         # First make sure separate morphemes are geez
    #         morphs2 = [[(g, f) for g in geezify_morph(m, alt=True)] for m, f in morphs]
    #     else:
    #         morphs2 = []
    #         for m, f in morphs:
    #             conv = convert_labial(m)
    #             morphs2.append([(c, f) for c in conv])
    #     morphs = allcombs(morphs2)
    # #    print(" Morphs {}".format(morphs))
    #     if not features:
    #         morphs = [[w[0] for w in word] for word in morphs]
    #     else:
    #         # Rejoin morpheme and features for each word
    #         morphs = [[''.join(m) for m in word] for word in morphs]
    #     return [sep.join(m) for m in morphs]
    #
    # def root2string(self, root):
    #     """If root contains '+', it consists of a root and a template, which need to be
    #     integrated."""
    #     if '+' in root:
    #         cons, temp = root.split('+')
    #         cons = segment(cons, self.language.seg_units)
    #         cons = [c for c in cons if c not in ['a', '_']]
    # #        print("root cons: {}, temp {}".format(cons, temp))
    #         if 'tt' in temp:
    #             temp = temp.replace('tt', 't_')
    #         temp = [(int(t) if t.isdigit() else t) for t in temp]
    #         form = []
    #         last_cons = ''
    #         for index, t in enumerate(temp):
    #             if isinstance(t, int):
    #                 # Template positions are 1-based, not 0-based
    #                 c = cons[t-1]
    #                 if c != last_cons:
    #                     form.append(cons[t-1])
    #                 else:
    #                     # Identical consonants; geminate
    #                     form.append('_')
    #                 last_cons = c
    #             elif index == 0 and t in "aeiouIE":
    #                 form.append("'" + t)
    #             elif t in "stm":
    #                 form.append(t)
    #                 last_cons = t
    #             else:
    #                 form.append(t)
    #                 # A vowel or _ character was added so clear the last consonant
    #                 last_cons = ''
    #         return '{' + ''.join(form) + '}'
    #     else:
    #         return '{' + root + '}'
