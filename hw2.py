"""
HW 2 CODE

Please write all your code for HW 2 in this file and include this file
in your submission.
"""
import os
import re
import sys
from pathlib import Path
from typing import Optional, Union

import biberplus.tagger.tagger
import biberplus.tagger.tagger_utils
import nltk
from biberplus.tagger import tag_text
from bs4 import BeautifulSoup
from nltk.corpus import CorpusReader
from nltk.probability import FreqDist

""" Fix the biberplus bug for Windows users """

warning_shown = False


def build_variable_dictionaries() -> dict[str, set[str]]:
    global warning_shown
    if not warning_shown:
        print("You are running Sophie's modified version of biberplus, which "
              "which fixes compatibility issues with Windows.",
              file=sys.stderr)
        warning_shown = True

    script_dir = Path(os.path.dirname(biberplus.tagger.tagger_utils.__file__))
    constant_files = script_dir.glob("constants/*.txt")
    variables_dict = {}

    for constant_file in constant_files:
        # E.g. constants/suasive_verbs.txt -> suasive_verbs
        file_name = constant_file.stem
        variables_dict[file_name] = \
            biberplus.tagger.tagger_utils.read_in_variables(constant_file)

    return variables_dict


biberplus.tagger.tagger.build_variable_dictionaries = \
    build_variable_dictionaries

""" Problem 1: Python Exercises """


def function_without_implementation():
    raise NotImplementedError("This function has no implementation.")


""" Problem 2: Analysis of Inaugural Addresses """


def get_encodings(corpus: CorpusReader) -> dict[str, str]:
    """
    Problem 2c: Identifies the encoding for each file in an NLTK corpus.
    This function assumes that the encoding of a file is utf8 if it
    contains at least one token ending in "â" when loaded in latin1 en-
    coding. Otherwise, this function assumes that the file's encoding is
    latin1.

    :param corpus: An NLTK corpus
    :return: A dict mapping each file in corpus.fileids() to its en-
        coding (either "utf8" or "latin1")
    """
    raise NotImplementedError("get_encodings has not been implemented yet.")


def feature_dist(text: str) -> FreqDist:
    """
    Problem 2f: Counts the number of occurrences of each grammatical
    feature tag assigned by biberplus's tag_text function to a corpus.
    See hw2-pset.ipynb for a usage example. See Appendix A of Alkiek et
    al. (2025) for the full list of tags:
    https://arxiv.org/abs/2502.18590v1

    :param text: A corpus, represented as untokenized plaintext
    :return: A FreqDist mapping each possible tag appearing in text to
        the number of times that tag occurs in tag_text(text)
    """
    raise NotImplementedError("feature_dist has not been implemented yet.")


""" Problem 3: Analysis of Nomination Acceptance Speeches """

# Download the default NLTK tokenizer if it's not installed already
nltk.download("punkt_tab")


def get_raw_text_from_html(filename: str) -> str:
    """
    Problem 3a: Extracts the text of an HTML file containing a nomina-
    tion acceptance speech from the American Presidency Project website.
    Paragraphs of the text (represented by p elements in the HTML code)
    should be separated by '\n'.

    :param filename: The name of an HTML file containing a speech
    :return: The text of the speech
    """
    raise NotImplementedError("get_raw_text_from_html has not been "
                              "implemented yet.")


def preprocess(raw_text: str) -> str:
    """
    Removes substrings of the form "[...]" from a string.

    :param raw_text: A string
    :return: raw_text, with anything between square brackets removed
    """
    return re.sub(r"\[.*]", "", raw_text)


class ConventionSpeechCorpus:
    """
    A container for nomination acceptance speeches represented as HTML
    files. This class is a simplified version of the NLTK CorpusReader.
    """

    def __init__(self, filename_or_folder: Optional[str] = None):
        """
        When a ConventionSpeechCorpus object is constructed, this method
        initializes it by loading one or more HTML files into it. More
        HTML files can be added to the ConventionSpeechCorpus using the
        .add method.

        :param filename_or_folder: An HTML file to be loaded into the
            corpus or a folder containing HTML files, all of which will
            be loaded into the corpus
        """
        self._raw_texts = dict()
        if filename_or_folder.endswith(".html"):
            self.add(filename_or_folder)
        elif filename_or_folder is not None:
            """
            Problem 3c: In this branch of the if-else block, you can
            assume that filename_or_folder is the name of a folder. 
            Please add all HTML files in this folder to self._raw_texts.
            """
            raise NotImplementedError(
                "Initialization from a directory has not been "
                "implemented yet.")

    def add(self, filename: str):
        raw = preprocess(get_raw_text_from_html(filename))
        self._raw_texts[filename] = raw

    """ NLTK Corpus Methods """

    def fileids(self) -> list[str]:
        return list(self._raw_texts.keys())

    def raw(self, files: Union[str, list[str], None] = None) -> str:
        """
        Problem 3d: Returns the raw text from self._raw_texts for one or
        more files in the corpus, depending on the value of the files
        parameter.
        - If files is the name of a file, then the raw text for that
          file is returned.
        - If files is a list of files, then the raw text for those files
          is returned, in the same order that the files appear in files.
        - If files is None, then the raw text for all files is returned,
          in lexicographic order of their filenames (Python's default
          sort order for the filenames).
        - If files contains at least one file not represented in
          self._raw_texts, this method should raise a ValueError.
        If raw texts are returned for more than one file, they should be
        separated by '\n'.

        :param files: The name of a file, or a list of files
        :return: The raw text for the files(s) described by files. If
            files is None, then the raw text for all files is returned
        """
        raise NotImplementedError(
            "ConventionSpeechCorpus.raw has not been implemented yet.")

    def words(self, files: Union[str, list[str], None] = None) -> list[str]:
        """
        Problem 3d: Returns text for one or more files in the corpus,
        preprocessed using the preprocess function and tokenized using
        the nltk.word_tokenize function. Which files are included in the
        tokenized text depends on the value of the files parameter.
        - If files is the name of a file, then the tokens for the text
          from that file is returned.
        - If files is a list of files, then the tokens for the text in
          those files is returned, in the same order that the files
          appear in files.
        - If files is None, then the tokens for the text in all files is
          returned, in lexicographic order of their filenames (Python's
          default sort order for the filenames).
        - If files contains at least one file not represented in
          self._raw_texts, this method should raise a ValueError.
        If lists of tokens are returned for more than one file, they
        should be concatenated.
        
        :param files: The name of a file, or a list of files
        :return: The tokenized text from the files(s) described by
            files. If files is None, then the tokenized text for all
            files is returned
        """
        raise NotImplementedError(
            "ConventionSpeechCorpus.words has not been implemented yet.")
