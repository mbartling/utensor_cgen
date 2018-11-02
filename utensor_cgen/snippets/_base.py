# -*- coding:utf8 -*-
from abc import ABCMeta
from copy import deepcopy

from .template_env import env as _env

__all__ = ["Snippet", "SnippetContainerBase"]


class SnippetBase(object):
  __metaclass__ = ABCMeta
  __core_template_name__ = None
  __headers__ = None

  # Optional
  __var_decls_template__ = None
  __var_create_template__ = None
  __var_destroy_template__ = None

  def __init__(self):
    if self.__core_template_name__ is None:
      raise ValueError('No {}.__core_template_name__ not defined'.format(type(self)))
    if self.__headers__ is None:
      raise ValueError('No {}.__headers__ not defined'.format(type(self)))
    if not isinstance(self.__headers__, set):
      raise ValueError('__headers__ should be of type set, get {}'.format(type(self.__headers__)))
    self.template_vars = {}

  @property
  def template_name(self):
    return self.__core_template_name__

  @property
  def template(self):
    if self.__core_template_name__ is None:
      raise ValueError('No template name: please override class attribute __core_template_name__')
    return _env.get_template(self.__core_template_name__)
  
  @property
  def vars_decls_template(self):
    if self.__var_decls_template__ is None:
      raise ValueError('No template name: please override class attribute __var_decls_template__')
    return _env.get_template(self.__var_decls_template__)
  
  @property
  def vars_create_template(self):
    if self.__var_create_template__ is None:
      raise ValueError('No template name: please override class attribute __var_create_template__')
    return _env.get_template(self.__var_create_template__)
  
  @property
  def vars_destroy_template(self):
    if self.__var_destroy_template__ is None:
      raise ValueError('No template name: please override class attribute __var_destroy_template__')
    return _env.get_template(self.__var_destroy_template__)

  @property
  def headers(self):
    if self.__headers__ is None:
      raise ValueError('No __headers__ found, all snippet class should overwrite this class attribute')
    return deepcopy(self.__headers__)

  def add_header(self, header):
    self.__headers__.add(header)

  def remove_header(self, header):
    self.__headers__.remove(header)


class Snippet(SnippetBase):  # pylint: W0223

  def render(self):
    return self.template.render(**self.template_vars)
  def render_vars_decls(self):
    return self.vars_decls_template.render(**self.template_vars)
  def render_vars_create(self):
    return self.vars_create_template.render(**self.template_vars)
  def render_vars_destroy(self):
    return self.vars_destroy_template.render(**self.template_vars)


class SnippetContainerBase(SnippetBase):

  def __init__(self, snippets=None):
    SnippetBase.__init__(self)

    if snippets is None:
      snippets = []
    self._snippets = snippets
    for snp in self._snippets:
      self.__headers__.update(snp.headers)

  def add_snippet(self, snippet):
    """Add snippet into containers
    """
    if not isinstance(snippet, Snippet):
      msg = "expecting Snippet object, get {}".format(type(snippet))
      raise TypeError(msg)
    self.__headers__.update(snippet.headers)
    self._snippets.append(snippet)

  def render(self):
    return self.template.render(snippets=self._snippets, **self.template_vars)
