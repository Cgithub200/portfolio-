using System.ComponentModel.DataAnnotations;
using System.ComponentModel.DataAnnotations.Schema;

namespace VertoTest.Models
{
    public class StaticContent
    {

        [Key]
        [DatabaseGenerated(DatabaseGeneratedOption.Identity)]
        public int ContentId { get; set; }
        
       
        public string? StaticClass { get; set; }

        public string? Title { get; set; }
        public string? Body { get; set; }

        public int Order { get; set; }

        public ICollection<Image>? Images { get; set; }

        public DateTime DateUpdated { get; set; }

        [ForeignKey("SliderItem")]
        public int? SliderItemID { get; set; }

        public ContentPanels? SliderItem { get; set; }
    }
}
