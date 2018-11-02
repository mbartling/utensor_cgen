#ifndef __{{graph_name}}_HPP__
#define __{{graph_name}}_HPP__

class {{graph_name}} : /*public uTensorModel*/ {
  private:
    {% for snippet in snippets %}
    {{snippet.render_vars_decls()}}
    {% endfor %}

    Context ctx;

  public:
  
    {{graph_name}}();
    ~{{graph_name}}();

    {%if placeholders%}
    S_TENSOR infer({%for ph in placeholders%}Tensor* input_{{loop.index0}}{%if not loop.last %},{%endif%}{%endfor%});
    {% else %}
    S_TENSOR infer();
    {% endif %}

    Context& get_ctx(void);

}

#endif
