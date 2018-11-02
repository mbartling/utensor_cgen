#ifndef __{{graph_name}}_HPP__
#define __{{graph_name}}_HPP__

class {{graph_name}} : /*public uTensorModel*/ {
  private:
    {% for snippet in snippets%}
    {% for (outputvar, output_dtype) in zip(snippet.template_vars["output_vars"], snippet.template_vars["out_dtypes"]) %}
    RamTensor<output_dtype>* outputvar;
    {% endfor %}
    {% endfor %}

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
